"""Go2Env: a batched locomotion environment for Genesis. Used by 05_genesis_go2.py.

This is step 3's PointReachEnv idea taken to its conclusion. There, one env held one
numpy state and step() returned one obs. Here ONE object holds N robots, and every
buffer is a torch tensor with a leading dimension of N. step() takes an (N, 12) action
tensor and returns (N, 45) observations. There is no loop over environments anywhere:
resets, rewards, and command sampling are all masked tensor ops. This is the shape
every GPU-native simulator (Isaac Lab, MJX, Genesis) uses, and why they are fast.

The interface is rsl-rl's VecEnv, not Gymnasium:
    reset()   -> obs                                   obs is a TensorDict {"policy": (N, 45)}
    step(a)   -> obs, reward (N,), done (N,), extras   extras["time_outs"] marks truncations

Adapted from Genesis' examples/locomotion/go2_env.py. Logic is unchanged; comments added.

Observation (45 floats), all in the robot's own frame so the policy is heading-invariant:
    base angular velocity (3), gravity direction (3, = "which way is down", i.e. tilt),
    velocity command (3), joint pos offset from default (12), joint vel (12), last action (12)
Action (12): target joint angle offsets. A PD controller (kp, kd) turns them into torques,
exactly like the real Go2 firmware, so the trained policy runs on the real robot as-is.

Reward: mostly "match the commanded velocity", with small penalties for bouncing, jerky
actions, straying from the default pose, and wrong body height. Scales are in the cfg.
"""

import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    """Uniform samples in [lower, upper], one per env. lower/upper are 1-D tensors."""
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, camera=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg  # rsl-rl's logger reads env.cfg to record it alongside the run
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # 50 Hz control, like the real robot. Each control step runs `substeps` physics steps.
        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # real robot has 1 step of latency
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        # --- Scene: the Genesis equivalent of a MuJoCo XML. Built once, then batched N ways.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # Feet-on-ground only, so ~20 pairs is plenty. Lower = less memory per env.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),  # only draw env 0
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        # Optional offscreen camera for watch_go2.py. Must be added before build().
        self.cam = None
        if camera:
            self.cam = self.scene.add_camera(res=(960, 540), pos=(2.0, -2.0, 1.2), lookat=(0.0, 0.0, 0.3), fov=40, GUI=False)

        # This is the "run N copies" call. Everything after it is batched.
        self.scene.build(n_envs=num_envs)

        # Joint names -> DOF indices. The first 6 DOFs are the free-floating base (xyz + rotation);
        # the 12 leg joints come after, hence slice(6, 18) in step().
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD gains. The policy outputs target angles; the sim (and the real robot) does
        # torque = kp * (target - pos) - kd * vel.
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state, stored once and broadcast into reset envs.
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # --- Buffers. Every one has a leading num_envs dimension. Allocated once, updated in place.
        N = self.num_envs
        self.base_lin_vel = torch.empty((N, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((N, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((N, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((N,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((N,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((N,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((N, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                command_cfg["lin_vel_x_range"],
                command_cfg["lin_vel_y_range"],
                command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((N, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((N, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((N, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((N, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # rsl-rl reads "time_outs" and "episode" from here

        # Reward terms are methods named _reward_<key>. Scales are multiplied by dt so the
        # per-episode sum is independent of control frequency.
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((N,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _resample_commands(self, envs_idx):
        """New random velocity command for the masked envs (or all, if envs_idx is None)."""
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()  # all N envs advance at once

        # Read back state. Velocities are rotated into the body frame.
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Reward: a weighted sum of terms, each an (N,) tensor.
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # New command every resampling_time_s, staggered per env by episode length.
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # Done = timed out OR tipped over OR the solver blew up in that env.
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # rsl-rl bootstraps the value at time-outs but not at falls (Gymnasium's truncated vs terminated).
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset only the done envs, in place, with masks. No Python loop.
        self._reset_idx(self.reset_buf)

        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        # rsl-rl 5 takes observation groups; we have one, "policy", shared by actor and critic.
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        """Reset the envs where envs_idx is True (all envs if None)."""
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            m = envs_idx[:, None]
            torch.where(m, self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(m, self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(m, self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(m, self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(m, 0.0)
            self.base_ang_vel.masked_fill_(m, 0.0)
            self.dof_vel.masked_fill_(m, 0.0)
            self.actions.masked_fill_(m, 0.0)
            self.last_actions.masked_fill_(m, 0.0)
            self.last_dof_vel.masked_fill_(m, 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Per-term episode averages for the envs that just ended. rsl-rl logs these to
        # TensorBoard as Episode/rew_<term>, which is how you see WHICH reward term is moving.
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # --- Reward terms. Each returns an (N,) tensor; the scale (sign included) lives in the cfg.
    def _reward_tracking_lin_vel(self):
        # exp(-error/sigma): 1.0 when matching the commanded xy velocity, falls off smoothly.
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Bouncing. Penalized.
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Jerky actions. Penalized; matters a lot for real hardware.
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Stray far from the standing pose. Penalized.
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Crouching or standing tall. Penalized (scale -50, the strongest term).
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
