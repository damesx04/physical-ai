"""Step 5 — Genesis. Thousands of robots on the GPU, one policy, rsl-rl's PPO.

    python 05_genesis_go2.py                          # 4096 envs on Metal, 100 iterations (~2.5 min: 1 min compile, 1.5 min train)
    python 05_genesis_go2.py --benchmark              # just measure env-steps/s at --num-envs
    python 05_genesis_go2.py --cpu --num-envs 256     # CPU backend, for comparison
    tensorboard --logdir runs                         # runs/genesis_go2 sits next to the cheetah runs
    python watch_go2.py --record                      # video of the trained policy

Go2 is a real quadruped (Unitree). The task: walk forward at 0.5 m/s without falling.
The policy this trains is the same architecture Unitree-class labs deploy to hardware.

What changed vs steps 1-4, and why:

1. The simulator is batched. MuJoCo ran 4 cheetahs in 4 processes and shipped numpy arrays
   to PPO. Genesis builds ONE scene with N robots and steps them all in a single GPU kernel
   launch. Obs, rewards, and dones are torch tensors that never leave the GPU. Measured on
   this M4 Pro (zero-action steps, env-steps/s = num_envs x steps/s):

       num_envs     Metal (gs.gpu)     CPU (gs.cpu)
             64          9,900           36,900
            256         37,500                -
          1,024        130,900           32,900
          4,096        335,400                -

   Two lessons in that table. The CPU is FASTER at small N: a kernel launch on Metal costs
   more than 64 robots' worth of physics. And the CPU is flat at ~35k no matter what: it is
   already saturated, while Metal keeps scaling. Step 4's HalfCheetah run got ~4k env-steps/s
   end to end. This is 80x that. (An RTX 4090 gets ~10x more again; the 43M FPS headline
   number is a 4090 with a simpler scene.) First build compiles Metal kernels: ~60s, every run.

2. The env is a class you own (go2_env.py). No Gymnasium registry has "Go2 walking". You write
   the observation, reward, termination, and reset yourself, as batched tensor ops. Read it;
   the reward terms are where you will spend your time on any real robot.

3. The RL library is rsl-rl, not SB3. SB3 wants numpy VecEnvs and would copy every batch
   GPU->CPU->GPU. rsl-rl (ETH Zurich's, used by Isaac Lab) takes torch tensors directly and its
   PPO is written for "few steps per env, huge batch": 24 steps x 4096 envs = 98k samples per
   update. Compare SB3's 2048 x 4 = 8k. The tradeoff: less API polish, no EvalCallback, and
   you read its TensorBoard tags (Train/mean_reward, Episode/rew_*) instead of eval/mean_reward.
"""

import argparse
import os
import pickle
import shutil
import time

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=100, help="PPO updates; each is num_steps_per_env x num_envs samples")
p.add_argument("--cpu", action="store_true", help="gs.cpu instead of gs.gpu (Metal). Slower at scale, faster to build")
p.add_argument("--benchmark", action="store_true", help="time zero-action steps at --num-envs and exit")
p.add_argument("--run-name", default="genesis_go2")
p.add_argument("--seed", type=int, default=1)
args = p.parse_args()

LOG_DIR = f"runs/{args.run_name}"  # rsl-rl writes TensorBoard events AND model_<it>.pt here

# --- Configs. These dicts ARE the environment spec; watch_go2.py reloads them from cfgs.pkl.
env_cfg = dict(
    num_actions=12,
    # Standing pose, radians. The policy outputs OFFSETS from this, scaled by action_scale.
    default_joint_angles={
        "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    },
    joint_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ],
    kp=20.0,  # PD gains, same as the real robot's low-level controller
    kd=0.5,
    termination_if_roll_greater_than=10,  # degrees. Tip past this = fell = episode over
    termination_if_pitch_greater_than=10,
    base_init_pos=[0.0, 0.0, 0.42],
    base_init_quat=[1.0, 0.0, 0.0, 0.0],
    episode_length_s=20.0,
    resampling_time_s=4.0,  # new velocity command every 4 s
    action_scale=0.25,
    simulate_action_latency=True,
    clip_actions=100.0,
)
obs_cfg = dict(obs_scales=dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05))
# The scales ARE the reward function. Positive = want more, negative = penalty. Try changing
# base_height to 0 and watch the gait change; that is the whole job on a real robot.
reward_cfg = dict(
    tracking_sigma=0.25,
    base_height_target=0.3,
    reward_scales=dict(
        tracking_lin_vel=1.0,
        tracking_ang_vel=0.2,
        lin_vel_z=-1.0,
        base_height=-50.0,
        action_rate=-0.005,
        similar_to_default=-0.1,
    ),
)
# Walk forward at exactly 0.5 m/s. Widen these ranges (e.g. x in [-1, 1], yaw in [-1, 1]) to
# train a policy you can steer with a joystick; it takes more iterations.
command_cfg = dict(num_commands=3, lin_vel_x_range=[0.5, 0.5], lin_vel_y_range=[0, 0], ang_vel_range=[0, 0])

# rsl-rl's PPO config. Note vs SB3 defaults: num_steps_per_env=24 (SB3: 2048) because the
# batch comes from width, not length; adaptive LR schedule targeting a KL of 0.01; ELU 512-256-128
# actor and critic (SB3: tanh 64-64).
train_cfg = {
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    },
    "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 24,
    "save_interval": 25,  # model_25.pt, model_50.pt, ... plus model_<last>.pt
    "run_name": args.run_name,
    "logger": "tensorboard",
}

# --- Backend. gs.gpu resolves to Metal on Apple Silicon (torch device "mps"), CUDA elsewhere.
gs.init(
    backend=gs.cpu if args.cpu else gs.gpu,
    precision="32",
    logging_level="warning",
    seed=args.seed,
    performance_mode=True,
)

t0 = time.time()
env = Go2Env(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
print(f"built {args.num_envs} envs on {gs.device} in {time.time() - t0:.0f}s (kernel compile dominates)")

if args.benchmark:
    actions = torch.zeros((args.num_envs, env.num_actions), device=gs.device)
    for _ in range(20):  # warm-up; first steps include JIT
        env.step(actions)
    n_steps = 200
    t0 = time.time()
    for _ in range(n_steps):
        env.step(actions)
    elapsed = time.time() - t0
    print(f"num_envs={args.num_envs} backend={'cpu' if args.cpu else 'gpu'}: "
          f"{n_steps / elapsed:.0f} steps/s = {args.num_envs * n_steps / elapsed:,.0f} env-steps/s")
    raise SystemExit

# Fresh log dir each run (rsl-rl appends otherwise). Save the cfgs so watch_go2.py can rebuild
# an identical env. This is the Genesis equivalent of saving the VecNormalize .pkl.
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
with open(f"{LOG_DIR}/cfgs.pkl", "wb") as f:
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

runner = OnPolicyRunner(env, train_cfg, LOG_DIR, device=gs.device)
# init_at_random_ep_len: stagger the episode clocks so all 4096 envs don't time out in lockstep.
t0 = time.time()
runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
elapsed = time.time() - t0

samples = args.iterations * train_cfg["num_steps_per_env"] * args.num_envs
print(f"\n{args.run_name}: {args.iterations} iterations, {samples:,} samples in {elapsed / 60:.1f} min "
      f"({samples / elapsed:,.0f} samples/s end to end, PPO updates included)")
print(f"saved {LOG_DIR}/model_{args.iterations - 1}.pt + cfgs.pkl (rsl-rl counts from 0); TensorBoard events in the same dir")
print(f"watch it:  python watch_go2.py --run-name {args.run_name} --record")
