"""
IronStride Custom Gymnasium Environment for Unitree H1 Humanoid.

Wraps MuJoCo simulation of the Unitree H1 biped with a dense, multi-objective
reward function and configurable domain randomization. Designed for
benchmarking PPO vs SAC in high-dimensional continuous control.

Author: Shriman Raghav Srinivasan
Project: SR-HUM-04-ADV
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Locate the Unitree H1 MJCF model
# ---------------------------------------------------------------------------

def _find_h1_scene_xml() -> str:
    """Return the absolute path to the Unitree H1 scene.xml from mujoco_menagerie."""
    try:
        import mujoco_menagerie
        menagerie_root = Path(mujoco_menagerie.__path__[0])
    except (ImportError, AttributeError):
        # Fallback: look for a local clone
        menagerie_root = Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie"

    scene_xml = menagerie_root / "unitree_h1" / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(
            f"Could not find Unitree H1 scene.xml at {scene_xml}. "
            "Install mujoco_menagerie or clone it into third_party/."
        )
    return str(scene_xml)


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML configuration."""
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# IronStrideEnv
# ═══════════════════════════════════════════════════════════════════════════

class IronStrideEnv(gym.Env):
    """
    Custom Gymnasium environment for the Unitree H1 humanoid in MuJoCo.

    Observation Space  (R^n):
        - Base linear velocity (3)
        - Base angular velocity (3)
        - Projected gravity vector in body frame (3)
        - Joint positions (n_joints)
        - Joint velocities (n_joints)
        - Previous action (n_actuators)

    Action Space (R^n_actuators):
        - Normalised PD position target offsets in [-1, 1],
          scaled to per-joint radian limits at step time.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config_path: Optional[str] = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()

        # Configuration
        self.cfg = _load_config(config_path)
        env_cfg = self.cfg["env"]
        reward_cfg = self.cfg["reward"]
        dr_cfg = self.cfg["domain_randomization"]

        # ── MuJoCo model & data ─────────────────────────────────────
        scene_xml = _find_h1_scene_xml()
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # Physics timestep config
        self.model.opt.timestep = env_cfg["simulation_dt"]
        self._n_substeps = env_cfg["n_substeps"]
        self._control_dt = env_cfg["control_dt"]

        # ── Dimensions (derived from model) ─────────────────────────
        self.n_joints = self.model.njnt - 1  # exclude the free-floating root joint
        self.n_actuators = self.model.nu
        self._action_scale = self._compute_action_scale()

        # ── Spaces ──────────────────────────────────────────────────
        #   obs = [lin_vel(3), ang_vel(3), proj_grav(3),
        #          qpos_joints(n_joints), qvel_joints(n_joints-1+6? no...),
        #          prev_action(n_actuators)]
        #
        # Joint positions exclude the free-floating root (7 qpos for freejoint).
        # Joint velocities exclude the free-floating root (6 qvel for freejoint).
        n_qpos_joints = self.model.nq - 7   # subtract free-joint quaternion pos
        n_qvel_joints = self.model.nv - 6   # subtract free-joint vel
        self._obs_dim = 3 + 3 + 3 + n_qpos_joints + n_qvel_joints + self.n_actuators

        high = np.inf * np.ones(self._obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_actuators,),
            dtype=np.float32,
        )

        # ── Reward parameters ───────────────────────────────────────
        self._cmd_vel = env_cfg["command_velocity"]
        self._w_track = reward_cfg["w_tracking"]
        self._w_posture = reward_cfg["w_posture"]
        self._w_energy = reward_cfg["w_energy"]
        self._w_symmetry = reward_cfg["w_symmetry"]
        self._w_survival = reward_cfg["w_survival"]
        self._w_height = reward_cfg.get("w_height", 2.0)
        self._target_height = reward_cfg.get("target_height", 0.98)
        self._tracking_sigma = reward_cfg["tracking_sigma"]

        # Pre-compute max torque for energy normalization
        if self.model.actuator_ctrlrange.any():
            ctrl_max = np.abs(self.model.actuator_ctrlrange).max(axis=1)
        else:
            ctrl_max = np.ones(self.n_actuators)
        self._torque_norm = float(np.sum(ctrl_max ** 2))  # for normalising R_energy

        # ── Termination ─────────────────────────────────────────────
        self._min_height = env_cfg["min_torso_height"]
        self._max_tilt = env_cfg["max_torso_tilt"]

        # ── Domain randomization ────────────────────────────────────
        self._dr_enabled = dr_cfg["enabled"]
        self._friction_range = dr_cfg["friction_range"]
        self._mass_offset_range = dr_cfg["mass_offset_range"]
        self._perturbation_cfg = dr_cfg["perturbation"]

        # ── Internal state ──────────────────────────────────────────
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._step_count = 0

        # Store default/nominal values for domain randomization resets
        self._default_friction = self.model.geom_friction.copy()
        # Find the torso body for mass randomization
        self._torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if self._torso_body_id == -1:
            # Try alternative names
            for name in ["torso", "trunk", "pelvis"]:
                self._torso_body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if self._torso_body_id != -1:
                    break
        self._default_torso_mass = (
            self.model.body_mass[self._torso_body_id]
            if self._torso_body_id >= 0
            else 0.0
        )

        # ── Rendering ──────────────────────────────────────────────
        self.render_mode = render_mode
        self._renderer = None
        self._render_width = width
        self._render_height = height

        # Identify left/right actuator indices for symmetry reward
        self._left_actuator_ids, self._right_actuator_ids = (
            self._identify_lr_actuators()
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._step_count = 0

        # Apply domain randomization
        if self._dr_enabled:
            self._apply_domain_randomization()

        # Forward kinematics so sensor data is valid
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Clear any lingering external forces (fixes perturbation persistence bug)
        self.data.xfrc_applied[:] = 0.0

        # Scale normalised action → PD position targets
        pd_targets = self._action_scale * action
        self.data.ctrl[:] = pd_targets

        # Step physics (n_substeps × simulation_dt = control_dt)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Mid-episode perturbation
        if (
            self._dr_enabled
            and self._perturbation_cfg["enabled"]
            and self.np_random.random() < self._perturbation_cfg["probability"]
        ):
            self._apply_impulse_perturbation()

        # Observation, reward, termination
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = False  # handled by max_episode_steps wrapper
        info = self._get_info()

        # Cache action for next step's observation
        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self._render_height, self._render_width
            )

        self._renderer.update_scene(self.data)
        frame = self._renderer.render()

        if self.render_mode == "rgb_array":
            return frame
        return frame  # 'human' mode — caller displays

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observation Construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector from MuJoCo state."""
        # Base (free-joint) orientation quaternion → rotation matrix
        quat = self.data.qpos[3:7]
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)

        # Linear velocity in body frame
        world_linvel = self.data.qvel[0:3]
        body_linvel = rot.T @ world_linvel

        # Angular velocity in body frame
        world_angvel = self.data.qvel[3:6]
        body_angvel = rot.T @ world_angvel

        # Projected gravity in body frame (world gravity = [0, 0, -9.81])
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = rot.T @ gravity_world

        # Joint positions and velocities (exclude free-joint)
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        obs = np.concatenate([
            body_linvel.astype(np.float32),           # (3,)
            body_angvel.astype(np.float32),            # (3,)
            projected_gravity.astype(np.float32),      # (3,)
            joint_pos.astype(np.float32),              # (n_qpos_joints,)
            joint_vel.astype(np.float32),              # (n_qvel_joints,)
            self._prev_action,                         # (n_actuators,)
        ])
        return obs

    # ------------------------------------------------------------------
    # Reward Computation
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Multi-objective reward:
            R_total = w1·R_track + w2·R_posture + w3·R_energy
                    + w4·R_symmetry + w5·R_survival + w6·R_height
        """
        # --- R_track: Gaussian on forward velocity error ---
        forward_vel = self.data.qvel[0]  # x-axis velocity in world frame
        vel_error = forward_vel - self._cmd_vel
        r_track = np.exp(-vel_error**2 / (2 * self._tracking_sigma**2))

        # --- R_posture: Exponential penalty on torso tilt ---
        quat = self.data.qpos[3:7]
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        # Torso z-axis should align with world z-axis
        # cos(tilt) = dot(body_z, world_z) = rot[2,2]
        cos_tilt = np.clip(rot[2, 2], -1.0, 1.0)
        tilt_angle = np.arccos(cos_tilt)
        r_posture = np.exp(-5.0 * tilt_angle**2)

        # --- R_energy: Normalised penalty on sum of squared torques ---
        #   Scaled to [0, -1] range so it doesn't dominate other terms
        torques = self.data.actuator_force
        r_energy = -np.sum(torques**2) / max(self._torque_norm, 1.0)

        # --- R_symmetry: Normalised L/R velocity variance ---
        r_symmetry = 0.0
        if len(self._left_actuator_ids) > 0 and len(self._right_actuator_ids) > 0:
            n_pairs = min(len(self._left_actuator_ids), len(self._right_actuator_ids))
            left_vel = self.data.actuator_velocity[self._left_actuator_ids[:n_pairs]]
            right_vel = self.data.actuator_velocity[self._right_actuator_ids[:n_pairs]]
            diff = left_vel - right_vel
            r_symmetry = -np.mean(diff ** 2)  # normalised by count

        # --- R_survival: Constant per-step bonus ---
        r_survival = 1.0

        # --- R_height: Reward for maintaining standing height ---
        torso_z = self.data.qpos[2]
        height_error = torso_z - self._target_height
        r_height = np.exp(-5.0 * height_error**2)

        # Weighted sum
        reward = (
            self._w_track * r_track
            + self._w_posture * r_posture
            + self._w_energy * r_energy
            + self._w_symmetry * r_symmetry
            + self._w_survival * r_survival
            + self._w_height * r_height
        )

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self) -> bool:
        """Episode terminates if the humanoid falls or tilts too far."""
        # Torso height (z-position of the free-floating base)
        torso_z = self.data.qpos[2]
        if torso_z < self._min_height:
            return True

        # Torso tilt
        quat = self.data.qpos[3:7]
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        cos_tilt = np.clip(rot[2, 2], -1.0, 1.0)
        tilt_angle = np.arccos(cos_tilt)
        if tilt_angle > self._max_tilt:
            return True

        return False

    # ------------------------------------------------------------------
    # Domain Randomization
    # ------------------------------------------------------------------

    def _apply_domain_randomization(self) -> None:
        """Apply randomizations at episode reset."""
        # --- Floor friction ---
        friction_scale = self.np_random.uniform(*self._friction_range)
        # Scale all geom frictions by a random factor
        self.model.geom_friction[:] = self._default_friction * friction_scale

        # --- Torso mass perturbation ---
        if self._torso_body_id >= 0:
            mass_offset = self.np_random.uniform(*self._mass_offset_range)
            self.model.body_mass[self._torso_body_id] = (
                self._default_torso_mass + mass_offset
            )

    def _apply_impulse_perturbation(self) -> None:
        """Apply a lateral impulse to the torso (mid-episode disturbance)."""
        if self._torso_body_id < 0:
            return

        magnitude = self._perturbation_cfg["force_magnitude"]
        # Random lateral direction (x-y plane)
        angle = self.np_random.uniform(0, 2 * np.pi)
        force = np.array([
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            0.0,   # no vertical force
            0.0, 0.0, 0.0,  # no torque
        ])
        self.data.xfrc_applied[self._torso_body_id] = force

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_action_scale(self) -> np.ndarray:
        """
        Compute per-actuator scaling from normalised [-1,1] to radian limits.
        Uses the actuator control range if defined, otherwise defaults to ±π/2.
        Multiplied by 0.3 to prevent extreme initial torques.
        """
        ACTION_SCALE_FACTOR = 0.3  # conservative scaling to tame exploration
        if self.model.actuator_ctrlrange.any():
            low = self.model.actuator_ctrlrange[:, 0]
            high = self.model.actuator_ctrlrange[:, 1]
            scale = (high - low) / 2.0 * ACTION_SCALE_FACTOR
        else:
            scale = np.full(self.n_actuators, np.pi / 2 * ACTION_SCALE_FACTOR, dtype=np.float64)
        return scale.astype(np.float32)

    def _identify_lr_actuators(self) -> tuple[list[int], list[int]]:
        """
        Identify left- and right-side actuator indices by name convention.
        Unitree H1 joints follow: left_hip_yaw, right_hip_yaw, etc.
        """
        left_ids: list[int] = []
        right_ids: list[int] = []

        for i in range(self.n_actuators):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name is None:
                continue
            name_lower = name.lower()
            if "left" in name_lower or name_lower.startswith("l_"):
                left_ids.append(i)
            elif "right" in name_lower or name_lower.startswith("r_"):
                right_ids.append(i)

        return left_ids, right_ids

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary info dict."""
        return {
            "torso_height": float(self.data.qpos[2]),
            "forward_velocity": float(self.data.qvel[0]),
            "step_count": self._step_count,
        }
