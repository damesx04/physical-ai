# Physical AI learning plan

My path from zero to robot learning, one small script at a time. I started with reinforcement
learning for locomotion and moved on to imitation learning for manipulation. Each step adds one new
idea. Each script opens with a docstring that says what changed from the previous step and why.

Everything runs in simulation on a MacBook (Apple Silicon). Training times and benchmark numbers in
the docstrings were measured on an M4 Pro.

## The ladder

### Reinforcement learning (`RL/`)

| Step | Script | New idea |
|---|---|---|
| 1 | `01_cartpole.py` | The basic loop: make env, build model, learn, evaluate (PPO, Stable-Baselines3) |
| 2 | `02_reacher.py` | Continuous actions on a MuJoCo arm, same code |
| 3 | `03_point_env.py` | Writing your own Gymnasium environment and reward |
| 4 | `04_halfcheetah.py` | Locomotion: observation/reward normalization, longer runs, tuned hyperparameters |
| 5 | `05_genesis_go2.py`, `go2_env.py` | Thousands of Unitree Go2 quadrupeds in parallel on the GPU (Genesis + rsl-rl) |

`watch.py` and `watch_go2.py` replay trained policies in a window or record them to video.

### Imitation learning (`imitation/`)

The plan for this half is in [`imitation/IMITATION_SPEC.md`](imitation/IMITATION_SPEC.md).

| Step | Script | New idea |
|---|---|---|
| 6 | `06_pusht_diffusion.py` | Behavior cloning with Diffusion Policy on Push-T |
| 7 | `07_aloha_act.py` | Image observations, two arms, action chunking (ACT on ALOHA sim) |

## Running it

Each script lists its own commands at the top. Rough setup:

- **Steps 1–4:** `pip install "stable-baselines3[extra]" "gymnasium[mujoco]"`
- **Step 5:** `pip install genesis-world rsl-rl-lib`
- **Steps 6–7:** [LeRobot](https://github.com/huggingface/lerobot) with its Push-T and ALOHA sim
  extras. Use a separate virtual environment, because LeRobot pins its own PyTorch version.

Checkpoints, TensorBoard runs and videos are written to `checkpoints/`, `runs/` and `videos/`.
They are git-ignored. The two small `halfcheetah_*_vecnormalize.pkl` files are committed: they
hold the normalization stats `watch.py` needs to replay the step 4 policies.
