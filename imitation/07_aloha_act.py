"""Step 7 — ACT on ALOHA sim transfer cube. Image observations, a bimanual action space, and action chunking.

    python 07_aloha_act.py                                          # train on lerobot/aloha_sim_transfer_cube_scripted, LeRobot default ACT, ~9 h here
    python 07_aloha_act.py --chunk 1 --run-name aloha_act_chunk1    # ablation: no chunking, watch compounding error
    python 07_aloha_act.py --eval checkpoints/aloha_act/last        # 50 rollouts: success rate, video
    python 07_aloha_act.py --eval checkpoints/aloha_act/last --temporal-ensemble   # paper-style eval: query every step, blend overlapping chunks
    python 07_aloha_act.py --inspect                                # what a training sample looks like: one image in, 100 actions out
    tensorboard --logdir runs                                       # runs/aloha_act next to runs/pusht_u256

ALOHA transfer cube: two ViperX arms on a table, one cube. Right arm picks the cube up, hands it to
the left arm in mid-air. Observation is one 480x640 RGB camera ("top") plus 14 joint positions
(6 joints + 1 gripper per arm). Action is 14 joint-position targets at 50 Hz. Episodes are 400 steps
(8 s). Reward is a stage counter 0..4 (touched, lifted, both grippers on it, transferred), and success
is reward == 4 at any point. The dataset (lerobot/aloha_sim_transfer_cube_scripted) is 50 episodes /
20,000 frames from a scripted planner, so the demos are clean and unimodal; the ACT paper's harder
setting uses 50 human demos of the same task.

What changed vs step 6 (Diffusion Policy on Push-T), and why:

1. Pixels to joint targets. Push-T handed the policy a 96x96 image and asked for a 2D point. Here the
   path is: 480x640 image -> ResNet18 -> 15x20x512 feature map -> flatten to 300 tokens -> plus one
   token for the 14-D joint state and one for the CVAE latent -> 4-layer transformer encoder -> 1-layer
   decoder with 100 learned query positions -> linear head -> 100 x 14 joint targets. The env's PD
   controllers track those targets. Nothing in the policy knows what a gripper is.

2. Action chunking is the main idea, not a detail. Diffusion Policy predicted 16 actions and used 8.
   ACT predicts 100 (2 s at 50 Hz) and executes all 100 before looking again. The point is
   compounding error: a per-step policy that drifts a little off the demo distribution sees a state it
   never trained on, drifts more, and so on. Predicting a whole chunk from one observation makes the
   effective horizon 100x shorter in decisions, so there are 100x fewer chances to compound.
   `--chunk 1` turns that off; same model, same data, and it should fail visibly.

3. A CVAE, not a diffusion model, handles multimodality. ACT trains a second transformer (the CVAE
   encoder) that sees the ground-truth action chunk and compresses it into a 32-D latent z. The policy
   is conditioned on z during training, so it can explain away "style" variation in the demos instead
   of averaging it. At eval z is set to zero (the prior mean), so you get the average style. That is
   cheaper than 100 denoising steps but weaker: it cannot pick between two genuinely different modes
   at test time. With scripted demos there is only one mode, so it does not matter here; the KL term
   (weight 10) mostly keeps z from being used at all. Watch train/kld_loss stay near zero.

4. Loss is L1 on the chunk, masked where the chunk runs past the episode end. The dataset returns
   `action_is_pad` for those positions; the loss ignores them. This is why there is no
   EpisodeAwareSampler / drop_n_last_frames here: the last frame of an episode is a valid sample with
   99 padded actions.

5. Eval is success rate over rollouts, and that number is all you get. The reward is a 0..4 stage
   counter, so eval/avg_max_reward tells you how far the policy typically got (2 = lifted, 3 = both
   grippers on the cube). A reward curve across training does not exist: training never touches the
   env. The ACT paper reports ~86% (scripted) / ~50% (human) on transfer cube; LeRobot's own checkpoint
   on the human data is around 50%.

Config: LeRobot's ACTConfig defaults, which are the paper's ALOHA settings: chunk_size=100,
n_action_steps=100, ResNet18 backbone (ImageNet init), dim_model 512, 4 encoder / 1 decoder layers,
CVAE with latent 32, kl_weight 10, dropout 0.1, AdamW lr 1e-5 (constant, no scheduler), weight decay
1e-4, batch 8, 100k steps. The paper evaluates with temporal ensembling (query every step, exponentially
weight the overlapping chunks); LeRobot's default is no ensembling, and `--temporal-ensemble` turns it
on at eval only. Both are reported below once measured.

Measured on this M4 Pro (24 GB), MPS, batch 8: 52M params, 1.7 GB MPS, ~3.1 it/s with synthetic data,
so 100k steps is ~9 h if video decoding keeps up (480x640 frames, one per sample). The model is small
enough that this does not swap with a browser open, unlike step 6's UNet. An ALOHA episode is 400
MuJoCo steps at ~11 ms each plus one policy call per 100 steps, so 50 eval episodes is ~5 min.

Where things go (all relative to this file):
    runs/<run-name>/            TensorBoard: train/loss, train/l1_loss, train/kld_loss, eval/pc_success, eval/avg_max_reward
    checkpoints/<run-name>/     <step>/ dirs with model.safetensors + config.json + processor jsons,
                                `last` and `best` symlinks, run_args.json, final_eval.json
    videos/<run-name>/          eval_episode_*.mp4 from the final 50-episode eval
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

DEFAULT_DATASET = "lerobot/aloha_sim_transfer_cube_scripted"
ALOHA_TASK = "AlohaTransferCube-v0"

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--dataset", default=DEFAULT_DATASET, help="Hub repo id (lerobot/aloha_sim_transfer_cube_human also works)")
p.add_argument("--run-name", default="aloha_act")
p.add_argument("--steps", type=int, default=100_000, help="gradient steps (LeRobot default 100k)")
p.add_argument("--batch-size", type=int, default=8, help="LeRobot default 8 for ACT")
p.add_argument("--chunk", type=int, default=100, help="chunk_size = n_action_steps. 100 = paper. 1 = the ablation")
p.add_argument("--num-workers", type=int, default=4, help="dataloader workers decoding 480x640 video frames")
p.add_argument("--log-freq", type=int, default=100)
p.add_argument("--save-freq", type=int, default=5_000)
p.add_argument("--eval-freq", type=int, default=5_000, help="in-loop rollouts every N steps (0 = off)")
p.add_argument("--eval-episodes", type=int, default=10, help="episodes per in-loop eval")
p.add_argument("--final-eval-episodes", type=int, default=50)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--cpu", action="store_true", help="force CPU (default: MPS if available)")
p.add_argument("--eval", metavar="CKPT", help="evaluate a checkpoint dir and exit (no training)")
p.add_argument("--temporal-ensemble", action="store_true",
               help="with --eval: query the policy every step and blend overlapping chunks (coeff 0.01, as in the paper)")
p.add_argument("--n-videos", type=int, default=2, help="episodes to render to mp4 during --eval / final eval")
p.add_argument("--inspect", action="store_true", help="print dataset layout + one training sample and exit")
args = p.parse_args()


# ----------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------


def pick_device():
    import torch

    if not args.cpu and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_env_cfg():
    from lerobot.envs.configs import AlohaEnv

    # 480x640 top camera, 14-D agent_pos, 400-step episodes at 50 Hz: matches the dataset exactly.
    return AlohaEnv(task=ALOHA_TASK)


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
    """Each checkpoint is ~200 MB. Keep only the dirs `last` and `best` point at."""
    keep = {os.readlink(ckpt_root / n) for n in ("last", "best") if (ckpt_root / n).is_symlink()}
    for d in ckpt_root.iterdir():
        if d.is_dir() and not d.is_symlink() and d.name.isdigit() and d.name not in keep:
            shutil.rmtree(d)


def load_checkpoint(ckpt: str, device: str, temporal_ensemble: bool = False):
    """Rebuild policy + processors from a dir written by save_checkpoint (or any LeRobot pretrained dir)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    ckpt = Path(ckpt).resolve()
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    if temporal_ensemble:
        # Paper-style eval: the model still predicts a 100-chunk, but we ask it every step and average the
        # chunks that overlap at each timestep with weights exp(-0.01 * age). Same weights, different rollout.
        cfg.temporal_ensemble_coeff = 0.01
        cfg.n_action_steps = 1
    policy = make_policy(cfg, env_cfg=make_env_cfg())
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def make_policy_cfg(device: str):
    from lerobot.policies.act import ACTConfig

    return ACTConfig(chunk_size=args.chunk, n_action_steps=args.chunk, device=device)


# ----------------------------------------------------------------------------------------------------
# --inspect: what an episode looks like on disk, and what one training sample is
# ----------------------------------------------------------------------------------------------------


def inspect_dataset():
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(args.dataset)
    print(f"dataset: {args.dataset}  root: {meta.root}")
    print(f"episodes: {meta.total_episodes}  frames: {meta.total_frames}  fps: {meta.fps}  "
          f"tasks: {list(meta.tasks.index) if hasattr(meta.tasks, 'index') else meta.tasks}")
    print("\nfeatures (one row per frame in the parquet; the camera is decoded from mp4 on access):")
    for k, ft in meta.features.items():
        print(f"  {k:28s} {ft['dtype']:8s} {tuple(ft['shape'])}")
    cfg = make_policy_cfg("cpu")
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    print(f"\ndelta_timestamps (ACTConfig.action_delta_indices = range(chunk_size={cfg.chunk_size})):")
    for k, v in delta_timestamps.items():
        print(f"  {k}: {len(v)} offsets, {v[0]:.2f}s .. {v[-1]:.2f}s")
    ds = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps, video_backend="pyav")
    print("\non disk:")
    for sub in ("meta", "data", "videos"):
        d = Path(meta.root) / sub
        if not d.exists():
            continue
        files = sorted(x for x in d.rglob("*") if x.is_file())
        print(f"  {sub}/  ({len(files)} files)")
        for f in files[:4]:
            print(f"    {f.relative_to(meta.root)}  {f.stat().st_size / 1e6:.2f} MB")
        if len(files) > 4:
            print(f"    ... {len(files) - 4} more")

    ep_len = int(meta.episodes[0]["length"]) if "length" in meta.episodes[0] else 400
    for name, i in (("first frame of episode 0", 0), (f"last frame of episode 0 (index {ep_len - 1})", ep_len - 1)):
        s = ds[i]
        print(f"\n{name}, as the DataLoader returns it:")
        for k, v in s.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else ""
            dtype = getattr(v, "dtype", type(v).__name__)
            extra = ""
            if k == "action_is_pad":
                extra = f"  -> {int(v.sum())} of {len(v)} actions are padding (past the episode end, masked out of the loss)"
            print(f"  {k:28s} {str(dtype):14s} {shape}{extra}")
    print("\nnormalization (ACTConfig.normalization_mapping):")
    for ftype, mode in cfg.normalization_mapping.items():
        print(f"  {ftype:8s} -> {mode}")
    st = meta.stats["observation.state"]
    print("\njoint state stats (meta/stats.json): 14 = [L: waist shoulder elbow forearm_roll wrist_angle wrist_rotate gripper, R: same]")
    print(f"  mean={[round(float(x), 2) for x in st['mean']]}")
    print(f"  std ={[round(float(x), 2) for x in st['std']]}")
    print("  -> state and action are MEAN_STD normalized (not min-max like diffusion: no clip range to respect).")
    print("     Gripper std is tiny in scripted demos; a joint that never moves gets std ~0 and LeRobot clamps it.")
    print(f"\nimage tokens: 480x640 -> ResNet18 /32 -> {480 // 32}x{640 // 32} = {(480 // 32) * (640 // 32)} tokens of dim 512, "
          f"+1 state token +1 latent token = {(480 // 32) * (640 // 32) + 2} encoder inputs; {cfg.chunk_size} decoder queries")


# ----------------------------------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------------------------------


def train():
    import torch
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from torch.utils.tensorboard import SummaryWriter

    device = pick_device()
    set_seed(args.seed)
    run_dir = Path("runs") / args.run_name
    ckpt_root = Path("checkpoints") / args.run_name
    videos_dir = Path("videos") / args.run_name
    if ckpt_root.exists():
        sys.exit(f"{ckpt_root} exists; pick another --run-name or delete it.")

    cfg = make_policy_cfg(device)

    # Dataset. delta_timestamps makes ds[i] return the current frame plus the next `chunk_size` actions,
    # with `action_is_pad` marking positions past the episode end. return_uint8 keeps video frames as
    # bytes through the DataLoader; we /255 below.
    meta = LeRobotDatasetMetadata(args.dataset)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps, return_uint8=True,
                             video_backend="pyav")  # torchcodec needs system ffmpeg; pyav bundles it
    print(f"dataset {args.dataset}: {dataset.num_episodes} episodes, {dataset.num_frames} frames, fps {meta.fps}")
    print(f"action chunk: {len(delta_timestamps['action'])} steps = {len(delta_timestamps['action']) / meta.fps:.2f} s")

    policy = make_policy(cfg, ds_meta=dataset.meta)  # fills input/output features from the dataset
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: ACTPolicy {n_params / 1e6:.1f}M params on {device}, chunk {cfg.chunk_size}, "
          f"vae={cfg.use_vae}, backbone {cfg.vision_backbone}")

    opt_cfg = cfg.get_optimizer_preset()  # AdamW lr 1e-5, wd 1e-4; get_optim_params gives the backbone its own lr
    optimizer = opt_cfg.build(policy.get_optim_params())
    assert cfg.get_scheduler_preset() is None  # ACT: constant LR

    # Plain shuffled sampling over every frame: chunk padding handles episode ends (see docstring #4).
    workers = args.num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers > 0 else None,
        pin_memory=False,
        drop_last=False,
    )

    def batches():
        while True:
            yield from dataloader  # reshuffles per epoch

    import draccus

    run_args = {**vars(args), "device": device, "dataset_repo_id": args.dataset, "policy_config": draccus.encode(cfg)}
    writer = SummaryWriter(run_dir)
    writer.add_text("run_args", "```\n" + json.dumps(run_args, indent=2, default=str) + "\n```")
    camera_keys = dataset.meta.camera_keys
    best_success = -1.0
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

        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.log_freq == 0:
            dt = time.perf_counter() - t_log
            t_log = time.perf_counter()
            sps = args.log_freq / dt
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/l1_loss", loss_dict["l1_loss"], step)
            if "kld_loss" in loss_dict:
                writer.add_scalar("train/kld_loss", loss_dict["kld_loss"], step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/steps_per_s", sps, step)
            writer.add_scalar("train/dataloading_s", t_data, step)
            eta_h = (args.steps - step) / sps / 3600
            print(f"step {step:7d}  loss {loss.item():.4f}  l1 {loss_dict['l1_loss']:.4f}  "
                  f"kld {loss_dict.get('kld_loss', 0):.4f}  grad {grad_norm.item():.2f}  "
                  f"{sps:.2f} it/s  data {t_data:.2f}s  eta {eta_h:.1f} h", flush=True)

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
                  f"max-stage {agg['avg_max_reward']:.2f}/4  ({time.perf_counter() - t0:.0f}s)", flush=True)
            # Best = highest success rate; ties broken by how far the policy typically got.
            score = agg["pc_success"] + agg["avg_max_reward"] / 10
            if score > best_success:
                best_success = score
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
    policy, pre, post = load_checkpoint(args.eval, device, temporal_ensemble=args.temporal_ensemble)
    tag = "_ensemble" if args.temporal_ensemble else ""
    videos_dir = Path("videos") / (args.run_name + tag)
    t0 = time.perf_counter()
    info = run_eval(policy, pre, post, args.final_eval_episodes, args.seed, videos_dir=videos_dir, n_videos=args.n_videos)
    agg = info["aggregated"]
    out = Path(args.eval) / f"eval_{args.final_eval_episodes}ep_seed{args.seed}{tag}.json"
    out.write_text(json.dumps(info, indent=2, default=str))
    print(f"{args.final_eval_episodes} episodes from seed {args.seed}{' (temporal ensemble)' if tag else ''}: "
          f"success {agg['pc_success']:.0f}%  max-stage {agg['avg_max_reward']:.2f}/4  "
          f"sum-reward {agg['avg_sum_reward']:.1f}  ({time.perf_counter() - t0:.0f}s)")
    print(f"per-episode -> {out}\nvideos -> {videos_dir}")


if __name__ == "__main__":
    if args.inspect:
        inspect_dataset()
    elif args.eval:
        evaluate()
    else:
        train()
