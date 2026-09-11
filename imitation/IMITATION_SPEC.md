# Imitation Learning Ladder — Spec

Companion to steps 1–5 (RL for locomotion). This covers the other half of robot
learning: imitation learning for manipulation. Same rule as the RL ladder: one new
idea per step, a docstring at the top of each script saying what changed and why.

## Goal

Be able to take a demonstration dataset, train a manipulation policy on it, evaluate
it honestly, and know why it fails. No hardware required for the core path.

## Non-goals

- Domain randomization (only needed for sim-trained policies going to hardware; none here).
- Building a data-recording pipeline (LeRobot already has one).
- RL post-training (only if a step plateaus on something you care about; see Next steps).
- Perception, classical planning, ROS (separate pillars, separate ladder).

## Prerequisites

- Python 3.10+, a fresh venv separate from `.venv` here (LeRobot pins its own torch).
- `pip install lerobot` plus the sim extras it lists for pusht, aloha, and libero.
- Apple Silicon MPS works for Push-T and ALOHA sim. Step 3 wants a CUDA GPU
  (rent one for a day; ~$10–20).
- Check LeRobot's current examples first. The library moves fast; task and checkpoint
  names below may have shifted.

## Steps

### 06 — Diffusion Policy on Push-T

- **New idea:** behavior cloning, and a policy class that models a multimodal action
  distribution.
- **Do:** download the Push-T dataset (~200 human demos). Train Diffusion Policy with
  LeRobot's default config. Roll out 50 episodes, report success rate and mean coverage.
- **Also do (10 min):** record ~20 of your own demos with the mouse. Train on those
  alone, roll out, compare to the public data. Note what your demos did to the policy.
- **Done when:** success rate is in the range the Diffusion Policy paper reports (~0.9
  coverage on state obs) and you can explain why an MSE-trained MLP would average the
  demos into a push that works for nobody.
- **Write down:** what an episode looks like on disk, how actions are normalized, how
  a 2D task hides everything that makes real manipulation hard.

### 07 — ACT on ALOHA sim cube transfer

- **New idea:** image observations, bimanual action space, action chunking.
- **Do:** download the ALOHA sim transfer-cube dataset (50 scripted demos). Train ACT
  with LeRobot's default config. Roll out 50 episodes, report success rate.
- **Ablation (optional, one run):** set chunk size to 1 and watch it degrade. That is
  compounding error made visible.
- **Done when:** success rate is near the ACT paper's sim result and you can describe
  the path from camera pixels to 14 joint targets.
- **Write down:** why chunking helps, what the CVAE latent is for, how eval differs
  from RL (success rate over rollouts, not a reward curve).

### 08 — VLA fine-tune on LIBERO

- **New idea:** pretrained vision-language backbone, multi-task, language-conditioned.
- **Do:** fine-tune SmolVLA (or OpenVLA if you have the GPU) on one LIBERO suite
  (start with LIBERO-Spatial or LIBERO-Object, 10 tasks). Evaluate per-task success.
- **Done when:** the fine-tuned model beats the un-fine-tuned checkpoint on the suite
  and you can say which layers moved and what the target-robot demos taught it.
- **Write down:** what transferred from pretraining (perception, semantics) and what
  did not (action mapping). This is the cross-embodiment argument in miniature.
- **Budget:** one GPU-day. Do it once. The value is knowing the workflow, not mastery.

### 09 — DROID slice, offline

- **New idea:** real teleop data at scale, and why offline metrics are unsatisfying.
- **Do:** pull a slice of DROID (single-arm, one camera, one task family, a few
  hundred episodes). Play back 20 episodes and note what varies. Filter obvious junk.
  Train ACT or Diffusion Policy on the slice. Evaluate on held-out episodes by action
  prediction error.
- **Done when:** you have a trained policy and a written note on why held-out action
  error does not tell you whether it would succeed on a robot.
- **Budget:** two afternoons. If it stretches past that, stop; the lesson is already
  learned.

## Evaluation conventions (all steps)

- Success rate over ≥50 rollouts from random starts, with the seed recorded.
- Save the config next to the checkpoint so the run can be rebuilt (same habit as
  `cfgs.pkl` in step 5).
- Record one video per step in `videos/`.
- Log to `runs/` under the step name so TensorBoard sits beside the RL runs.

## Next steps (after 06–09)

Pick based on what failed, not on a list.

- **Hardware.** Two SO-100 arms (~$100 each) plus a webcam. LeRobot supports them
  natively: record 50 demos of one task, train ACT, deploy. This is the step nothing
  in sim replaces. Before buying, rehearse the exact workflow in LeRobot's SO-100 sim.
- **RL post-training.** If a policy plateaus below what you need: HIL-SERL (LeRobot
  has it) on the sim task, or advantage-weighted fine-tuning on your own rollouts.
- **Sim-to-real from the imitation side.** Generate demos in sim with randomized
  physics and visuals (MimicGen-style), clone, deploy on the SO-100. This is the only
  place domain randomization enters the imitation path.
- **DAgger.** Interactive corrections. Conceptually important, rarely used in current
  recipes; do it if compounding error is the failure you keep hitting.

## Reference

- Session notes: https://claude.ai/code/artifact/588e2cff-b324-4dab-a07f-8f8c0316f0cf
- Parallel sim / width vs length: https://claude.ai/code/artifact/22686333-cc2d-47a8-80cb-778144f49912
