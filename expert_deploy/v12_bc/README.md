# V12 simulation-dataset behavior cloning

This directory contains the V12 behavior-cloning implementation for five
simulated Go2 conditions:

- `g0`: healthy robot;
- `rr05`: right-rear calf at 50% strength;
- `p5`: 5 kg payload;
- `p75`: 7.5 kg payload;
- `rr03`: right-rear calf at 30% strength.

The datasets and trained BC policies are intentionally not stored in Git. Each
dataset has shape `[1000, 50, 45]` for observations and `[1000, 50, 12]` for
actions. Set the dataset path locally and run:

```bash
DATASET_PATH=/path/to/dataset.pt
OUTPUT_DIR=runs/v12_bc/g0

uv run python expert_deploy/v12_bc/train_bc_go2_dataset.py \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42 \
  --validation_fraction 0.2 \
  --epochs 300 \
  --batch_size 1024 \
  --learning_rate 0.0003 \
  --weight_decay 0.00001 \
  --hidden_dim 128 \
  --num_blocks 2 \
  --refit_epochs 100 \
  --device cuda:0
```

Training splits whole environment trajectories into 40 training trajectories
and 10 validation trajectories. `actor_validation_best.pt` is selected using
validation loss; `actor.pt` is the final policy after another 100 epochs of
refitting on all 50 trajectories.

## CENet behavior cloning

The CENet route estimates body-frame base linear velocity from the previous
10 proprioceptive observations and supplies the estimate plus a learned context
latent to the BC actor. Ground-truth velocity is read from `states[..., :3]`
only as an auxiliary training target; it is not used by the deployed actor.

```bash
uv run python expert_deploy/v12_bc/train_cenet_bc_go2_dataset.py \
  --dataset_path "$DATASET_PATH" \
  --output_dir runs/v12_bc/g0_cenet_h10 \
  --history_length 10 \
  --cenet_hidden_dim 128 \
  --latent_dim 16 \
  --velocity_loss_weight 1.0 \
  --epochs 300 \
  --refit_epochs 100 \
  --device cuda:0
```

Run a quantitative nominal-G0 rollout:

```bash
uv run python expert_deploy/v12_bc/evaluate_cenet_bc_go2_mjlab.py \
  --checkpoint runs/v12_bc/g0_cenet_h10/policy.pt \
  --dataset "$DATASET_PATH" \
  --output runs/v12_bc/g0_cenet_h10/mjlab_metrics.json \
  --num_envs 64 \
  --num_steps 1000 \
  --device cuda:0
```

The exported TorchScript and ONNX policies take `observation_history`
(`[batch, 10, 45]`) and `observation` (`[batch, 45]`) and return the 12-D
action and 3-D estimated body-frame base linear velocity.

The original E0 expert checkpoint used by the dataset collector is intentionally
not stored in Git. It is different from the later G0/D0 checkpoint used for
physical data collection.

## Recollect the simulation datasets

The repository pins the collector to the pre-DAgger implementation used by the
five-condition V12 pipeline. It does not contain expert action relabeling.
The pinned collector SHA256 is
`09a15c9bb6722e4bd76239b884bf0739fa3d0602aad148c0a3c10c8d0283adde`.

Provide the E0 checkpoint locally and run:

```bash
EXPERT=/path/to/e0_checkpoint \
DATA_ROOT="$PWD/runs/v12_e0_five_conditions_2m_to_50k" \
bash scripts/reinforcement_learning/rwm_dataset/collect_v12_e0_five_conditions_2m_to_50k_realrange.sh
```

For every condition, the launcher:

1. collects 2,048 independent 1,000-step simulated trajectories;
2. selects 50 complete trajectories;
3. audits the resulting 50,000-transition dataset and fails closed on an
   invalid result.
