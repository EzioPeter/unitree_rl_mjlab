# Go2 FlashSAC Expert: train, export, deploy, and collect

This is the minimal reproducible route used to produce the 45D Go2 expert that
was deployed on a physical Unitree Go2 and used to collect real-world CSV data.

It contains:

- the pinned `unitree_rl_mjlab` training base;
- the exact Go2 expert task and FlashSAC configuration;
- the original PyTorch checkpoint and verified ONNX policy;
- a pinned C++ deployment overlay;
- the controller configuration that logs physical-robot transitions to CSV.

It intentionally excludes the later RWM, TRACE, offline-policy, and ablation
experiments.

The separate V12 simulation-dataset behavior-cloning training code is
documented in
[`expert_deploy/v12_bc`](../expert_deploy/v12_bc/README.md). Its large
simulation datasets, source E0 expert, and trained BC policies are not committed
to Git.

## 1. Install

The tested platform is Linux with an NVIDIA GPU and a CUDA-enabled PyTorch
installation. The pinned physical-robot controller build is for x86_64 Linux;
its bootstrap script downloads and checksum-verifies ONNX Runtime 1.23.2.

Check out this branch, then install the repository normally:

```bash
git switch expert
uv sync
```

See the repository [README](../README.md) for the base simulator installation
notes.

Verify every included checkpoint and deployment artifact before use:

```bash
(cd expert_deploy && sha256sum --check SHA256SUMS)
```

## 2. Train the deployable expert

The reference run uses 1024 environments and 100 million environment steps:

```bash
bash scripts/go2_expert/train_expert.sh
```

For a short pipeline smoke test:

```bash
NUM_TRAIN_ENVS=16 NUM_ENV_STEPS=16384 \
OUTPUT_DIR="$PWD/runs/smoke" \
bash scripts/go2_expert/train_expert.sh
```

The actor consumes exactly 45 values:

```text
base angular velocity       3
projected gravity           3
velocity command            3
relative joint position    12
relative joint velocity    12
previous action            12
                           --
                           45
```

The critic sees the same values plus 3D base linear velocity. The action is a
12D normalized joint-position target.

## 3. Export ONNX

Export the included checkpoint:

```bash
bash scripts/go2_expert/export_checkpoint.sh
```

Or provide another checkpoint and output path:

```bash
bash scripts/go2_expert/export_checkpoint.sh \
  runs/go2_normal_flashsac_expert/step97656 \
  runs/go2_normal_flashsac_expert/step97656
```

The export script verifies PyTorch/ONNX numerical parity before accepting the
artifact.

The included checkpoint and deployed ONNX are byte-for-byte copies of the
original real-data-collection artifacts. A fresh stochastic RL run follows the
same training and deployment contract, but is not guaranteed to learn
bit-identical weights.

## 4. Prepare the physical-robot controller

```bash
bash expert_deploy/deployment/bootstrap_unitree_cpp_deploy.sh
cmake -S .local/unitree_cpp_deploy/deploy/robots/go2 \
      -B .local/unitree_cpp_deploy/deploy/robots/go2/build
cmake --build .local/unitree_cpp_deploy/deploy/robots/go2/build -j
```

Copy the prepared `.local/unitree_cpp_deploy` tree to the Go2 computer and
follow the upstream network/interface setup. The pinned controller config loads
the included `g0_d0_rrcalf_0p5` policy.

The robot must be suspended or have an immediate emergency stop available for
the first test. Confirm these values before enabling torque:

- control period: `0.02 s`;
- actor input: `45`;
- action size: `12`;
- policy joint order: `FL, FR, RL, RR`;
- hardware joint order: `FR, FL, RR, RL`;
- action scale and offset: the included `deploy.yaml`.

## 5. Collect real-world data

The included controller enables logging for the selected RL state. Each run
creates:

```text
logs/go2/g0_d0_rrcalf_0p5/logs/run_data_<timestamp>.csv
```

The CSV includes policy observations, raw/effective actions, desired and
measured joint state, torque, foot contact/force, command, episode step, and
safety fields. Stop the controller cleanly so buffered rows are flushed.

## Reference artifacts

- PyTorch checkpoint: `expert_deploy/artifacts/g0_d0_step97656/`
- Verified deployed ONNX:
  `expert_deploy/deployment/policies/g0_d0_rrcalf_0p5/exported/policy.onnx`
- Training config:
  `configs/flashsac_go2_normal_proprioceptive_expert.yaml`
- Deployment contract:
  `expert_deploy/deployment/policies/g0_d0_rrcalf_0p5/params/deploy.yaml`
- Exact provenance and invariants:
  [REPRODUCIBILITY.md](../expert_deploy/REPRODUCIBILITY.md)

## Provenance

The training base comes from
[`EzioPeter/unitree_rl_mjlab`](https://github.com/EzioPeter/unitree_rl_mjlab)
and retains its Apache-2.0 license. The deployment base is pinned to
[`wty-yy/unitree_cpp_deploy`](https://github.com/wty-yy/unitree_cpp_deploy);
its source is fetched by the bootstrap script and modified by the included
overlay.
