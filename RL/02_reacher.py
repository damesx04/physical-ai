"""Step 2 — Reacher with PPO. Same pattern as step 1, one string changed.

Reacher-v5: a two-joint arm in MuJoCo whose fingertip must reach a target dot.

What's new vs CartPole:
    action_space = Box(-1, 1, (2,))   # two joint torques, continuous
    obs_space    = Box(-inf, inf, (10,))  # cos/sin of joint angles, target xy,
                                          # joint velocities, fingertip - target vector
PPO handles continuous actions natively (Gaussian policy instead of categorical),
so nothing else in the code has to change.

Reward is NEGATIVE: -(distance to target) - (control cost). Episodes are 50 steps,
so a random policy scores ~-40 and a good one ~-4. Higher (closer to 0) is better.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

TIMESTEPS = 200_000

env = make_vec_env("Reacher-v5", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TIMESTEPS)
model.save("reacher_ppo")

eval_env = gym.make("Reacher-v5")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\nmean_reward={mean_reward:.1f} +/- {std_reward:.1f}  (~-4 = good, -40 = random)")
