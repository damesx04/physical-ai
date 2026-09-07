"""Step 4 — HalfCheetah with PPO. The step where "change one string" stops working.

    python 04_halfcheetah.py            # SB3 default hyperparameters (~4500 reward, ~4 min)
    python 04_halfcheetah.py --tuned    # rl-zoo tuned hyperparameters (~6200 reward, ~9 min)
    tensorboard --logdir runs           # watch both runs side by side

HalfCheetah-v5: a 2D cheetah with 6 joints that must run forward as fast as it can.
    action_space = Box(-1, 1, (6,))       # six joint torques
    obs_space    = Box(-inf, inf, (17,))  # joint angles + velocities. Note: NOT x position,
                                          # so the agent can't see how far it has run.
Reward is POSITIVE: forward velocity - 0.1 * control cost. Episodes are 1000 steps and
never terminate early (the cheetah can't "fall", it just flips and flails). Random policy
scores ~-300. At 1M steps on this machine: defaults ~4500, tuned ~6200. (The ~1500 number
often quoted for "PPO defaults on HalfCheetah" is what you get WITHOUT VecNormalize.)

Three things are new vs steps 1-3, and each exists because locomotion is harder:

1. VecNormalize. Joint velocities are ~1, body velocities ~10, rewards ~1 per step and
   ~1000 per episode. PPO's value function struggles with that spread. VecNormalize
   keeps running mean/std of obs and reward and rescales them. The stats are part of
   the model: they must be saved and reloaded alongside the .zip, otherwise the policy
   sees numbers on a different scale from the ones it trained on.

2. Scale. 1M steps instead of 200k. Training takes tens of minutes, so we log to
   TensorBoard and checkpoint instead of reading verbose=1 output. EvalCallback runs
   a fresh eval env every EVAL_EVERY steps and keeps the best model it has seen.

3. Hyperparameters. The defaults were tuned for Atari-ish problems. --tuned uses the
   rl-baselines3-zoo config for HalfCheetah. Run both and compare curves in TensorBoard:
   this is the clearest way to see what n_steps, batch_size, clip_range, and gamma do.
"""

import argparse
import os

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

p = argparse.ArgumentParser()
p.add_argument("--tuned", action="store_true", help="use rl-zoo hyperparameters instead of SB3 defaults")
p.add_argument("--timesteps", type=int, default=1_000_000)
p.add_argument("--n-envs", type=int, default=4)
args = p.parse_args()

ENV_ID = "HalfCheetah-v5"
RUN_NAME = "halfcheetah_tuned" if args.tuned else "halfcheetah_default"
EVAL_EVERY = 25_000  # in steps per env; x n_envs for total steps

# rl-baselines3-zoo hyperparams/ppo.yml, HalfCheetah entry. Found by Optuna, not by hand.
# Notable vs defaults: tiny learning rate, more epochs per batch, tighter clip range,
# a lower gamma (1000-step episodes, but the useful horizon is short), bigger network,
# and log_std_init=-2 so the initial policy is less noisy than the default std=1.
TUNED = dict(
    n_steps=512,
    batch_size=64,
    n_epochs=20,
    gamma=0.98,
    gae_lambda=0.92,
    learning_rate=2.0633e-05,
    ent_coef=0.000401762,
    clip_range=0.1,
    max_grad_norm=0.8,
    vf_coef=0.58096,
    policy_kwargs=dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)

# Training env. VecNormalize wraps the vec env, not each gym env.
env = VecNormalize(make_vec_env(ENV_ID, n_envs=args.n_envs), norm_obs=True, norm_reward=True)

# Eval env is its own VecNormalize so eval-time obs don't update the training stats.
# training=False freezes its stats; EvalCallback copies the training stats into it before
# each eval. norm_reward=False so logged eval rewards are in the env's real units.
eval_env = VecNormalize(make_vec_env(ENV_ID, n_envs=1), training=False, norm_obs=True, norm_reward=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,  # TensorBoard replaces the console spam
    tensorboard_log="runs",
    **(TUNED if args.tuned else {}),
)

callbacks = [
    # Best model lands in best/<run>/best_model.zip. Note it does NOT save VecNormalize
    # stats; use the final <run>_vecnormalize.pkl with it (stats barely move late in
    # training). Also writes eval/mean_reward to TensorBoard, which is the curve you
    # actually care about (train reward is normalized, so its scale means nothing).
    EvalCallback(
        eval_env,
        best_model_save_path=f"best/{RUN_NAME}",
        log_path=f"best/{RUN_NAME}",
        eval_freq=EVAL_EVERY,
        n_eval_episodes=5,
        deterministic=True,
    ),
    # Periodic snapshots so a killed run isn't a lost run. save_vecnormalize=True saves the
    # stats next to each checkpoint.
    CheckpointCallback(
        save_freq=EVAL_EVERY * 4,
        save_path=f"checkpoints/{RUN_NAME}",
        name_prefix="ppo",
        save_vecnormalize=True,
    ),
]

model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=RUN_NAME)

# Two files now, not one. watch.py needs both.
model.save(RUN_NAME)
env.save(f"{RUN_NAME}_vecnormalize.pkl")

# Final eval: fresh env, load the stats we just saved, freeze them.
final_env = VecNormalize.load(f"{RUN_NAME}_vecnormalize.pkl", make_vec_env(ENV_ID, n_envs=1))
final_env.training = False
final_env.norm_reward = False
mean_reward, std_reward = evaluate_policy(model, final_env, n_eval_episodes=10)
print(f"\n{RUN_NAME}: mean_reward={mean_reward:.0f} +/- {std_reward:.0f}  (~4500 default, ~6200 tuned, -300 random)")
print(f"saved {RUN_NAME}.zip + {RUN_NAME}_vecnormalize.pkl; best model in best/{RUN_NAME}/")
print(f"watch it:  python watch.py --env {ENV_ID} --model {RUN_NAME} --vecnorm {RUN_NAME}_vecnormalize.pkl --record")
