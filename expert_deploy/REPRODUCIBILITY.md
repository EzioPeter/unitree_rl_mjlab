# Reproducibility contract

This repository pins the training base to `EzioPeter/unitree_rl_mjlab`
commit `16dd94676e9403532fea785c4146b47828e40c28`.

The reference policy was trained with:

- seed `0`;
- `1024` parallel environments;
- `100,000,000` environment steps;
- task `Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert`;
- FlashSAC with a 45-dimensional actor observation and 48-dimensional critic
  observation;
- `calibrated_default` domain randomization;
- no payload and full-strength joints.

The reference checkpoint is `expert_deploy/artifacts/g0_d0_step97656`. Its exported ONNX is
deployed with a 20 ms control period, the joint mapping and action scaling in
`expert_deploy/deployment/policies/g0_d0_rrcalf_0p5/params/deploy.yaml`, and the
controller prepared by
`expert_deploy/deployment/bootstrap_unitree_cpp_deploy.sh`.

The deployment bootstrap pins `wty-yy/unitree_cpp_deploy` to commit
`9400a4a73eaa9f79e7e07f2df50061cc7fb7520c` and checksum-verifies the official
x86_64 ONNX Runtime 1.23.2 archive before extracting it.

Changing observation order, joint order, default joint pose, action scale, or
control period produces a different deployment contract.

The included checkpoint and deployed ONNX are byte-for-byte copies of the
original artifacts, as recorded in `SHA256SUMS`. A fresh training run
reproduces the same configuration, task, observation/action contract, export,
and deployment route. It is not expected to reproduce bit-identical learned
weights because GPU-parallel RL and simulation contain nondeterministic
operations.
