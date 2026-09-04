"""Step 1 — CartPole with PPO. The 'hello world' of RL.

The pattern here is the same one every later script uses:
    make env -> wrap in monitor -> build model -> learn -> evaluate.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

TIMESTEPS = 100_000

# 4 parallel copies of the env -> more samples per wall-clock second.
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TIMESTEPS)
model.save("cartpole_ppo")

# Fresh env for evaluation so training stats don't leak in.
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\nmean_reward={mean_reward:.1f} +/- {std_reward:.1f}  (500 = solved)")
