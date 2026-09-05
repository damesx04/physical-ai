"""Step 3 — Build your own environment. Same training code as steps 1-2.

PointReach: a dot on a 2D plane must move to a target. It is Reacher with the arm
removed, so the numbers are directly comparable (50-step episodes, negative-distance
reward, ~-40 random, ~-4 good).

A Gymnasium env is a class with:
    observation_space / action_space   what the agent sees / can do
    reset()  -> obs, info               start a new episode
    step(a)  -> obs, reward, terminated, truncated, info

PPO never sees the physics. It only sees these five things, which is why the
training code below is byte-for-byte the same as 02_reacher.py.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


class PointReachEnv(gym.Env):
    MAX_STEPS = 50
    SPEED = 0.1  # how far a max action moves the point in one step

    def __init__(self):
        super().__init__()
        # Action: desired velocity in x and y, each in [-1, 1]. Continuous, like Reacher.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Obs: [pos_x, pos_y, target_x, target_y, dx, dy]. Including target - pos is
        # redundant but makes the problem much easier for a small network.
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(6,), dtype=np.float32)
        self.pos = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def _obs(self):
        return np.concatenate([self.pos, self.target, self.target - self.pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        self.pos = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        self.target = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.pos = np.clip(self.pos + self.SPEED * action, -1.0, 1.0).astype(np.float32)
        self.steps += 1

        dist = float(np.linalg.norm(self.target - self.pos))
        ctrl_cost = 0.1 * float(np.square(action).sum())
        reward = -dist - ctrl_cost  # same shape as Reacher's reward

        terminated = False  # no failure state; the episode only ends on the time limit
        truncated = self.steps >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {"dist": dist}


# Catch shape/dtype/return-type mistakes before spending minutes training.
check_env(PointReachEnv())

TIMESTEPS = 100_000

env = make_vec_env(PointReachEnv, n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TIMESTEPS)
model.save("point_ppo")

eval_env = PointReachEnv()
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\nmean_reward={mean_reward:.1f} +/- {std_reward:.1f}  (~-4 = good, -40 = random)")
