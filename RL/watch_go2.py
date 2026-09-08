"""Watch a Genesis Go2 policy from 05_genesis_go2.py — recorded video or live viewer.

    python watch_go2.py --record                 # videos/go2_<run>.mp4, headless, reliable on macOS
    python watch_go2.py                          # live Genesis viewer (flaky on macOS; use --record)
    python watch_go2.py --ckpt 50 --seconds 20   # a specific checkpoint, longer clip

watch.py can't do this: the policy is an rsl-rl model, not an SB3 .zip, and the env is a
Genesis scene, not a Gymnasium one. Same idea though: rebuild the env exactly as trained
(from cfgs.pkl), load the weights, run deterministic actions, render. One env, CPU backend:
one robot doesn't need Metal and the CPU builds in half the time.

The printed number is velocity-tracking error, not reward. The command is 0.5 m/s forward;
a trained policy holds |v_x - 0.5| under ~0.1 m/s. "fell" means the roll/pitch termination
fired: the run keeps going after a reset, but you will see it in the video.
"""

import argparse
import glob
import os
import pickle
import re

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env

p = argparse.ArgumentParser()
p.add_argument("--run-name", default="genesis_go2")
p.add_argument("--ckpt", type=int, default=None, help="iteration number of model_<N>.pt; default = latest")
p.add_argument("--seconds", type=int, default=10)
p.add_argument("--record", action="store_true", help="write mp4 instead of opening a viewer")
p.add_argument("--outdir", default="videos")
args = p.parse_args()

log_dir = f"runs/{args.run_name}"
if args.ckpt is None:
    ckpts = [int(re.search(r"model_(\d+)\.pt", f).group(1)) for f in glob.glob(f"{log_dir}/model_*.pt")]
    args.ckpt = max(ckpts)
ckpt_path = f"{log_dir}/model_{args.ckpt}.pt"

with open(f"{log_dir}/cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

gs.init(backend=gs.cpu, precision="32", logging_level="warning")
env = Go2Env(
    num_envs=1,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
    show_viewer=not args.record,
    camera=args.record,
)

# rsl-rl has no PPO.load(): build the runner from the same train_cfg, then load weights into it.
runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
runner.load(ckpt_path, map_location=gs.device)
policy = runner.get_inference_policy(device=gs.device)  # deterministic: mean action, no noise
print(f"loaded {ckpt_path}")

if args.record:
    os.makedirs(args.outdir, exist_ok=True)
    out = f"{args.outdir}/go2_{args.run_name}_{args.ckpt}.mp4"
    env.cam.start_recording(save_to_filename=out, fps=int(1 / env.dt))

n_steps = int(args.seconds / env.dt)
vel_err, falls = 0.0, 0
obs = env.reset()
with torch.no_grad():
    for _ in range(n_steps):
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        vel_err += abs(float(env.base_lin_vel[0, 0]) - float(env.commands[0, 0]))
        falls += int(bool(dones[0]) and not bool(extras["time_outs"][0]))  # done but not a timeout = fell
        if args.record:
            # Chase cam: keep the robot in frame as it walks off.
            base = env.base_pos[0].tolist()
            env.cam.set_pose(pos=(base[0] + 2.0, base[1] - 2.0, 1.2), lookat=(base[0], base[1], 0.3))
            env.cam.render()

print(f"{args.seconds}s at cmd {float(env.commands[0, 0]):.2f} m/s: "
      f"mean |v_x - cmd| = {vel_err / n_steps:.3f} m/s, fell {falls}x  (trained: <0.1, 0)")

if args.record:
    env.cam.stop_recording()
    print(f"wrote {out}")
