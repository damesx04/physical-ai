"""Watch a trained policy — live window or recorded video.

    python watch.py                      # live window, CartPole
    python watch.py --record             # writes videos/*.mp4 instead
    python watch.py --env HalfCheetah-v5 --model halfcheetah_tuned \
        --vecnorm halfcheetah_tuned_vecnormalize.pkl --record

Never render during training; rendering every frame is ~100x slower.
Train headless, save, then run this.

--vecnorm: models trained under VecNormalize (step 4+) saw normalized observations.
Feed them raw ones and they flail. The .pkl holds the running mean/std; we load it and
normalize each obs the same way training did.
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

p = argparse.ArgumentParser()
p.add_argument("--env", default="CartPole-v1")
p.add_argument("--model", default="cartpole_ppo")
p.add_argument("--vecnorm", default=None, help="VecNormalize .pkl saved alongside the model")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--record", action="store_true", help="write mp4 instead of opening a window")
p.add_argument("--outdir", default="videos")
args = p.parse_args()

# "human" = real-time OS window. "rgb_array" = frames handed back as numpy,
# which is what RecordVideo needs to build the file.
env = gym.make(args.env, render_mode="rgb_array" if args.record else "human")

if args.record:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.outdir,
        name_prefix=args.env,
        # Only the episodes we asked for. A vec env auto-resets after the last one, which
        # would otherwise start a stub recording. Use `i % 10 == 0` to sample long runs.
        episode_trigger=lambda i: i < args.episodes,
    )

if args.vecnorm:
    # VecNormalize only wraps vec envs, so wrap the single env in a 1-env DummyVecEnv.
    # training=False freezes the stats; norm_reward=False keeps printed rewards in real units.
    env = VecNormalize.load(args.vecnorm, DummyVecEnv([lambda: env]))
    env.training = False
    env.norm_reward = False

model = PPO.load(args.model)

for ep in range(args.episodes):
    total, steps = 0.0, 0
    if args.vecnorm:
        # Vec env API: batched obs, no info from reset, done = terminated | truncated.
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += float(reward[0])
            steps += 1
            if done[0]:
                break
    else:
        obs, _ = env.reset()
        while True:
            # deterministic=True -> argmax, no exploration noise. What you want when watching.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
    print(f"episode {ep}: reward={total:.0f} steps={steps}")

env.close()
if args.record:
    print(f"\nwrote mp4s to {args.outdir}/")
