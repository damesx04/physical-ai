"""Step 6 — Diffusion Policy on Push-T. Behavior cloning, with a policy that can represent "either/or".

    python 06_pusht_diffusion.py                                   # train on lerobot/pusht (206 human demos), LeRobot default UNet (~23 h here)
    python 06_pusht_diffusion.py --unet 256 --run-name pusht_u256  # what fits this Mac: 76M-param UNet, ~8 h. No resume; pick the budget up front
    python 06_pusht_diffusion.py --eval checkpoints/pusht_diffusion/last   # 50 rollouts: success rate, coverage, video
    python 06_pusht_diffusion.py --record 20                       # drive the agent with the mouse, 20 demos -> data/pusht_mine
    python 06_pusht_diffusion.py --dataset data/pusht_mine --run-name pusht_mine   # train on your own demos only
    python 06_pusht_diffusion.py --inspect                         # what an episode looks like on disk + how actions are normalized
    tensorboard --logdir runs                                      # runs/pusht_diffusion; add --logdir ../RL/runs to see the RL curves too

Push-T: a 2D top-down task. A circular agent (you, or the policy) pushes a T-shaped block onto a
T-shaped target. Observation is a 96x96 RGB image plus the agent's (x, y). Action is the agent's
target (x, y) in pixel coordinates [0, 512]^2, tracked by a PD controller. Reward at every step is
"coverage": the fraction of the target T covered by the block. An episode succeeds at 95% coverage.
The dataset (lerobot/pusht) is 206 episodes / 25,650 frames at 10 Hz, recorded with a mouse by the
Diffusion Policy authors.

What changed vs the RL ladder (steps 1-5), and why:

1. No reward signal during training. Behavior cloning is supervised learning: given the last two
   observations, predict the next 16 actions the human took. The reward is only used at eval time
   to score rollouts. There is no exploration, no value function, no environment interaction until
   evaluation. Data quality IS the algorithm: the policy can only be as good as the demos.

2. The policy is a conditional denoiser, not a Gaussian. PPO's policy head outputs a mean and a
   std: one bump. Human demos are multimodal: from the same state, one demo goes around the block
   clockwise, another counter-clockwise. A Gaussian (or an MLP trained with MSE, which is the same
   thing) fits the AVERAGE of those, which pushes straight into the block and works for nobody.
   Diffusion Policy instead learns to turn noise into an action sequence, conditioned on the
   observation: sample noise, denoise 100 steps, get one of the modes. Different noise, different
   mode. That is the whole reason to pay for 100 forward passes per decision.

3. Action chunking. The model predicts a horizon of 16 actions and the env executes the first 8
   before asking again (n_action_steps). Fewer decisions, smoother trajectories, and less chance
   for small errors to compound step by step. Step 7 (ACT) makes this the main idea.

4. Normalization lives in a processor pipeline, not a VecNormalize wrapper. State and action are
   min-max scaled to [-1, 1] using dataset stats (so the diffusion's clip_sample range matches);
   images are ImageNet mean/std. The stats are saved next to the weights (policy_preprocessor.json)
   and must travel with them, same lesson as step 4's vecnormalize.pkl. `--inspect` prints them.

5. Eval is a success rate over N rollouts from seeded random starts, plus mean max-coverage. The
   paper reports ~0.84 coverage on image obs (their Table 1, Push-T image, 200 demos) and
   pc_success is a stricter number (>95% coverage). Training loss tells you almost nothing about
   either; you have to roll out.

Config: LeRobot's DiffusionConfig defaults, except the three values LeRobot's own Push-T checkpoint
and the paper use for this task: horizon=16, n_action_steps=8, crop_shape=(84, 84) random crop.
Everything else (ResNet18 + spatial softmax encoder, UNet 512/1024/2048, DDPM 100 steps, Adam 1e-4,
cosine LR with 500 warmup, batch 64) is the library default.

Measured on this M4 Pro (24 GB), MPS, batch 64:
    --unet 512 (LeRobot default, 263M params)   6.1 GB MPS   1.2 it/s   100k steps ~23 h
    --unet 256 (76M params)                     2.6 GB MPS   3.5 it/s   100k steps ~8 h
The 512 UNet only fits with nothing else open: with a browser and VS Code up, the machine swaps and the
run stalls in page-wait (I watched it do ~100 steps in 40 min). Gradient checkpointing does not help
because weights + Adam state dominate, not activations. The 256 UNet is what actually runs here, so the
first real run used `--unet 256`. Eval: ~70 s per batch of 10 episodes (100 DDPM steps every 8 env
steps), so the final 50-episode eval is ~6 min. Checkpoints are ~1 GB (512) / 0.3 GB (256); only `last`
and `best` are kept. Every dataset load pings the Hub for the dataset version; set HF_HUB_OFFLINE=1 once
it is cached if you are offline.

Where things go (all relative to this file):
    runs/<run-name>/            TensorBoard: train/loss, train/lr, eval/pc_success, eval/avg_max_reward
    checkpoints/<run-name>/     <step>/ dirs with model.safetensors + config.json + processor jsons,
                                `last` and `best` symlinks, run_args.json
    videos/<run-name>/          eval_episode_*.mp4 from the final 50-episode eval
    data/pusht_mine/            your own demos as a LeRobotDataset (--record)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

PUSHT_TASK = "Push the T-shaped block onto the T-shaped target."
DEFAULT_DATASET = "lerobot/pusht"

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--dataset", default=DEFAULT_DATASET, help="Hub repo id, or a local dir written by --record")
p.add_argument("--run-name", default="pusht_diffusion")
p.add_argument("--steps", type=int, default=100_000, help="gradient steps (LeRobot default 100k)")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--unet", choices=["512", "256"], default="512",
               help="UNet width: 512 = LeRobot default down_dims (512,1024,2048), 263M params, ~6 GB; "
                    "256 = (256,512,1024), 76M params, ~2.6 GB, ~3x faster")
p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the UNet (less memory, slower)")
p.add_argument("--num-workers", type=int, default=4, help="dataloader workers decoding video frames")
p.add_argument("--log-freq", type=int, default=100)
p.add_argument("--save-freq", type=int, default=5_000)
p.add_argument("--eval-freq", type=int, default=5_000, help="in-loop rollouts every N steps (0 = off)")
p.add_argument("--eval-episodes", type=int, default=10, help="episodes per in-loop eval")
p.add_argument("--final-eval-episodes", type=int, default=50)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--cpu", action="store_true", help="force CPU (default: MPS if available)")
p.add_argument("--eval", metavar="CKPT", help="evaluate a checkpoint dir and exit (no training)")
p.add_argument("--n-videos", type=int, default=2, help="episodes to render to mp4 during --eval / final eval")
p.add_argument("--record", type=int, metavar="N", help="record N mouse demos into --record-dir and exit")
p.add_argument("--record-dir", default="data/pusht_mine")
p.add_argument("--inspect", action="store_true", help="print dataset layout + normalization stats and exit")
args = p.parse_args()


# ----------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------


def pick_device():
    import torch

    if not args.cpu and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dataset(spec: str):
    """'lerobot/pusht' -> (repo_id, None). 'data/pusht_mine' (a dir) -> ('local/pusht_mine', Path)."""
    path = Path(spec)
    if path.is_dir():
        return f"local/{path.name}", path.resolve()
    return spec, None


def make_env_cfg():
    from lerobot.envs.configs import PushtEnv

    # The dataset images are 96x96; LeRobot's PushtEnv defaults to 384 for the *observation*, which
    # would hand the policy a different image than it trained on. Match the data.
    return PushtEnv(observation_width=96, observation_height=96)


def run_eval(policy, preprocessor, postprocessor, n_episodes, seed, videos_dir=None, n_videos=0, n_envs=10):
    """Roll out `n_episodes` from seeded starts. Returns eval_policy's info dict (per_episode has the seeds)."""
    import torch
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    env_cfg = make_env_cfg()
    envs = make_env(env_cfg, n_envs=min(n_envs, n_episodes), use_async_envs=False)
    vec_env = envs[env_cfg.type][0]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    try:
        with torch.no_grad():
            info = eval_policy(
                vec_env,
                policy,
                env_pre,
                env_post,
                preprocessor,
                postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=n_videos,
                videos_dir=Path(videos_dir) if videos_dir else None,
                start_seed=seed,
            )
    finally:
        close_envs(envs)
    return info


def save_checkpoint(ckpt_root: Path, step: int, policy, preprocessor, postprocessor, run_args: dict):
    d = ckpt_root / f"{step:07d}"
    d.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(d)  # model.safetensors + config.json
    preprocessor.save_pretrained(d)  # normalization stats etc.
    postprocessor.save_pretrained(d)
    (d / "run_args.json").write_text(json.dumps(run_args, indent=2, default=str))
    last = ckpt_root / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(d.name)
    prune_checkpoints(ckpt_root)
    return d


def prune_checkpoints(ckpt_root: Path):
    """Each checkpoint is ~1 GB (262M params). Keep only the dirs `last` and `best` point at."""
    keep = {os.readlink(ckpt_root / n) for n in ("last", "best") if (ckpt_root / n).is_symlink()}
    for d in ckpt_root.iterdir():
        if d.is_dir() and not d.is_symlink() and d.name.isdigit() and d.name not in keep:
            shutil.rmtree(d)


def load_checkpoint(ckpt: str, device: str):
    """Rebuild policy + processors from a dir written by save_checkpoint (or any LeRobot pretrained dir)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    ckpt = Path(ckpt).resolve()
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    policy = make_policy(cfg, env_cfg=make_env_cfg())
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


# ----------------------------------------------------------------------------------------------------
# --inspect: what an episode looks like on disk, and how actions are normalized
# ----------------------------------------------------------------------------------------------------


def inspect_dataset():
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.diffusion import DiffusionConfig

    repo_id, root = resolve_dataset(args.dataset)
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    meta = ds.meta
    print(f"dataset: {repo_id}  root: {ds.root}")
    print(f"episodes: {meta.total_episodes}  frames: {meta.total_frames}  fps: {meta.fps}  "
          f"tasks: {list(meta.tasks.index) if hasattr(meta.tasks, 'index') else meta.tasks}")
    print("\nfeatures (one row per frame in the parquet; video features are decoded from mp4 on access):")
    for k, ft in meta.features.items():
        print(f"  {k:24s} {ft['dtype']:8s} {tuple(ft['shape'])}")
    print("\non disk:")
    for sub in ("meta", "data", "videos"):
        d = Path(ds.root) / sub
        if not d.exists():
            continue
        files = sorted(x for x in d.rglob("*") if x.is_file())
        print(f"  {sub}/  ({len(files)} files)")
        for f in files[:4]:
            print(f"    {f.relative_to(ds.root)}  {f.stat().st_size / 1e6:.2f} MB")
        if len(files) > 4:
            print(f"    ... {len(files) - 4} more")
    ep0 = meta.episodes[0]
    print("\nepisode 0 metadata row (which parquet + which video chunk + frame range):")
    for k, v in ep0.items():
        if not str(k).startswith("stats/"):
            print(f"  {k}: {v}")
    print("\none frame as the DataLoader returns it (ds[0]):")
    for k, v in ds[0].items():
        shape = tuple(v.shape) if hasattr(v, "shape") else ""
        dtype = getattr(v, "dtype", type(v).__name__)
        print(f"  {k:24s} {str(dtype):14s} {shape}")
    cfg = DiffusionConfig()
    print("\nnormalization (DiffusionConfig.normalization_mapping):")
    for ftype, mode in cfg.normalization_mapping.items():
        print(f"  {ftype:8s} -> {mode}")
    print("\nstats used for it (meta/stats.json, aggregated across episodes):")
    for k in ("action", "observation.state"):
        s = meta.stats[k]
        print(f"  {k}: min={s['min'].tolist()} max={s['max'].tolist()} mean={[round(float(x), 1) for x in s['mean']]}"
              f" std={[round(float(x), 1) for x in s['std']]}")
    print("  -> action and state are MIN_MAX scaled to [-1, 1]; the diffusion model only ever sees that range")
    print("     (clip_sample_range=1.0). Images are MEAN_STD with ImageNet stats. Unnormalize happens in the")
    print("     postprocessor before the action reaches the env.")


# ----------------------------------------------------------------------------------------------------
# --record N: mouse teleop -> LeRobotDataset (gym-pusht's own teleop_agent is broken in 0.1.6)
# ----------------------------------------------------------------------------------------------------


def record_demos(n_episodes: int, out_dir: Path, action_source=None, max_steps=300):
    """Record demos. `action_source(obs, env, t) -> action | None` overrides the mouse (used by the smoke test).
    Mouse: move the cursor onto the blue agent to start an episode; the agent follows the cursor.
    Keys: SPACE = end episode (keep), R = discard and redo, Q = stop and save what you have."""
    import gymnasium as gym
    import numpy as np
    import pygame
    from lerobot.datasets import LeRobotDataset

    import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)

    if out_dir.exists():
        sys.exit(f"{out_dir} exists. Delete it or pass a different --record-dir.")
    features = {
        "observation.image": {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(f"local/{out_dir.name}", fps=10, features=features, root=out_dir, use_videos=True,
                              video_backend="pyav")

    # rgb_array + our own window: in render_mode="human" gym-pusht returns None for the pixel obs.
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
        observation_width=96, observation_height=96, visualization_width=512, visualization_height=512,
        max_episode_steps=max_steps,
    )
    interactive = action_source is None
    if interactive:
        pygame.init()
        window = pygame.display.set_mode((512, 512))
        pygame.display.set_caption("Push-T teleop: move mouse onto agent to start | SPACE end | R redo | Q quit")
        clock = pygame.time.Clock()

    def show():
        if interactive:
            frame = env.render()  # 512x512 HWC; window coords == action coords (pymunk y-down)
            window.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))
            pygame.display.flip()

    def keys():
        if not interactive:
            return None
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "q"
            if e.type == pygame.KEYDOWN:
                return {pygame.K_SPACE: "space", pygame.K_r: "r", pygame.K_q: "q"}.get(e.key)
        return None

    ep, seed = 0, args.seed
    print(f"recording up to {n_episodes} episodes into {out_dir}")
    while ep < n_episodes:
        obs, info = env.reset(seed=seed)
        seed += 1
        show()
        started, quit_all, discard = False, False, False
        frames = []
        for t in range(max_steps):
            key = keys()
            if key == "q":
                quit_all = True
                break
            if key == "r":
                discard = True
                break
            if key == "space" and started:
                break
            if action_source is not None:
                action = action_source(obs, env, t)
            else:
                mouse = np.array(pygame.mouse.get_pos(), dtype=np.float32)
                agent = obs["agent_pos"]
                if not started and np.linalg.norm(mouse - agent) > 30:
                    clock.tick(10)
                    continue  # wait until the cursor is on the agent, like the original demo tool
                action = np.clip(mouse, 0, 512)
            started = True
            frame = {
                "observation.image": obs["pixels"],
                "observation.state": obs["agent_pos"].astype(np.float32),
                "action": np.asarray(action, dtype=np.float32),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            frame["next.reward"] = np.array([reward], dtype=np.float32)
            frame["next.success"] = np.array([info["is_success"]])
            frames.append(frame)
            show()
            if interactive:
                clock.tick(10)
            if terminated or truncated:
                break
        if discard or len(frames) < 10:
            print(f"  episode discarded ({len(frames)} frames)")
            if quit_all:
                break
            continue
        for f in frames:
            f["task"] = PUSHT_TASK
            ds.add_frame(f)
        ds.save_episode()
        ep += 1
        print(f"  episode {ep}/{n_episodes}: {len(frames)} frames, success={bool(frames[-1]['next.success'][0])}, "
              f"final coverage={info['coverage']:.2f}")
        if quit_all:
            break
    env.close()
    if interactive:
        pygame.quit()
    ds.finalize()
    print(f"saved {ds.meta.total_episodes} episodes / {ds.meta.total_frames} frames to {out_dir}")
    print(f"train on them:  python 06_pusht_diffusion.py --dataset {out_dir} --run-name {out_dir.name}")
    return ds


# ----------------------------------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------------------------------


def train():
    import torch
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.diffusion import DiffusionConfig
    from lerobot.utils.random_utils import set_seed
    from torch.utils.tensorboard import SummaryWriter

    device = pick_device()
    set_seed(args.seed)
    repo_id, root = resolve_dataset(args.dataset)
    run_dir = Path("runs") / args.run_name
    ckpt_root = Path("checkpoints") / args.run_name
    videos_dir = Path("videos") / args.run_name
    if ckpt_root.exists():
        sys.exit(f"{ckpt_root} exists; pick another --run-name or delete it.")

    # Policy config: LeRobot defaults + the Push-T values (see docstring). crop 84 of 96 = random shift
    # augmentation at train time, center crop at eval.
    down_dims = {"512": (512, 1024, 2048), "256": (256, 512, 1024)}[args.unet]
    cfg = DiffusionConfig(horizon=16, n_action_steps=8, crop_shape=(84, 84), device=device,
                          down_dims=down_dims, gradient_checkpointing=args.grad_ckpt)
    cfg.drop_n_last_frames = cfg.horizon - cfg.n_action_steps - cfg.n_obs_steps + 1  # 7

    # Dataset. delta_timestamps makes ds[i] return a stack: 2 past obs and 16 future actions per index.
    # return_uint8 keeps video frames as bytes through the DataLoader (4x less IPC); we /255 below.
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, return_uint8=True,
                             video_backend="pyav")  # torchcodec needs system ffmpeg; pyav bundles it
    print(f"dataset {repo_id}: {dataset.num_episodes} episodes, {dataset.num_frames} frames, fps {meta.fps}")
    print(f"delta_timestamps: {delta_timestamps}")

    policy = make_policy(cfg, ds_meta=dataset.meta)  # fills input/output features from the dataset
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: DiffusionPolicy {n_params / 1e6:.1f}M params on {device}")

    opt_cfg = cfg.get_optimizer_preset()
    optimizer = opt_cfg.build(policy.get_optim_params())
    lr_scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)

    # Sample (episode, t) pairs so the 16-action window never runs past the end of an episode.
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=cfg.drop_n_last_frames,
        shuffle=True,
        seed=args.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    workers = args.num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers > 0 else None,
        pin_memory=False,
        drop_last=False,
    )

    def batches():
        while True:
            yield from dataloader  # sampler reshuffles per epoch

    import draccus

    run_args = {**vars(args), "device": device, "dataset_repo_id": repo_id, "dataset_root": root,
                "policy_config": draccus.encode(cfg)}
    writer = SummaryWriter(run_dir)
    writer.add_text("run_args", "```\n" + json.dumps(run_args, indent=2, default=str) + "\n```")
    camera_keys = dataset.meta.camera_keys
    best_cov = -1.0
    t_log = time.perf_counter()
    dl_iter = batches()
    policy.train()
    print(f"training {args.steps} steps, batch {args.batch_size}, log -> {run_dir}")
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        for k in camera_keys:
            if k in batch and batch[k].dtype == torch.uint8:
                batch[k] = batch[k].float() / 255.0
        batch = preprocessor(batch)  # rename -> batch dim -> device -> normalize
        t_data = time.perf_counter() - t0

        loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if step % args.log_freq == 0:
            dt = time.perf_counter() - t_log
            t_log = time.perf_counter()
            sps = args.log_freq / dt
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/steps_per_s", sps, step)
            writer.add_scalar("train/dataloading_s", t_data, step)
            eta_h = (args.steps - step) / sps / 3600
            print(f"step {step:7d}  loss {loss.item():.4f}  grad {grad_norm.item():.2f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.1e}  {sps:.2f} it/s  data {t_data:.2f}s  eta {eta_h:.1f} h", flush=True)

        is_last = step == args.steps
        if step % args.save_freq == 0 or is_last:
            d = save_checkpoint(ckpt_root, step, policy, preprocessor, postprocessor, run_args)
            print(f"saved {d}")

        if (args.eval_freq and step % args.eval_freq == 0) or is_last:
            n_eps = args.final_eval_episodes if is_last else args.eval_episodes
            t0 = time.perf_counter()
            policy.eval()
            info = run_eval(policy, preprocessor, postprocessor, n_eps, args.seed,
                            videos_dir=videos_dir if is_last else None, n_videos=args.n_videos if is_last else 0)
            policy.train()
            agg = info["aggregated"]
            writer.add_scalar("eval/pc_success", agg["pc_success"], step)
            writer.add_scalar("eval/avg_max_reward", agg["avg_max_reward"], step)
            writer.add_scalar("eval/avg_sum_reward", agg["avg_sum_reward"], step)
            print(f"eval @ {step}: {n_eps} episodes  success {agg['pc_success']:.0f}%  "
                  f"coverage(max) {agg['avg_max_reward']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if agg["avg_max_reward"] > best_cov:
                best_cov = agg["avg_max_reward"]
                d = save_checkpoint(ckpt_root, step, policy, preprocessor, postprocessor, run_args)
                best = ckpt_root / "best"
                if best.is_symlink() or best.exists():
                    best.unlink()
                best.symlink_to(d.name)
                prune_checkpoints(ckpt_root)
            if is_last:
                (ckpt_root / "final_eval.json").write_text(json.dumps(info, indent=2, default=str))
                print(f"videos -> {videos_dir}   per-episode seeds + rewards -> {ckpt_root / 'final_eval.json'}")
    writer.close()


def evaluate():
    device = pick_device()
    policy, pre, post = load_checkpoint(args.eval, device)
    videos_dir = Path("videos") / args.run_name
    t0 = time.perf_counter()
    info = run_eval(policy, pre, post, args.final_eval_episodes, args.seed, videos_dir=videos_dir, n_videos=args.n_videos)
    agg = info["aggregated"]
    out = Path(args.eval) / f"eval_{args.final_eval_episodes}ep_seed{args.seed}.json"
    out.write_text(json.dumps(info, indent=2, default=str))
    print(f"{args.final_eval_episodes} episodes from seed {args.seed}: success {agg['pc_success']:.0f}%  "
          f"coverage(max) {agg['avg_max_reward']:.3f}  sum-reward {agg['avg_sum_reward']:.1f}  "
          f"({time.perf_counter() - t0:.0f}s)")
    print(f"per-episode -> {out}\nvideos -> {videos_dir}")


if __name__ == "__main__":
    if args.inspect:
        inspect_dataset()
    elif args.record:
        record_demos(args.record, Path(args.record_dir))
    elif args.eval:
        evaluate()
    else:
        train()
