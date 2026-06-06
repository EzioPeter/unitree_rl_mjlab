"""Evaluation metrics for Go2 RWM-U dynamics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from scripts.reinforcement_learning.rwm.dynamics import SystemDynamicsEnsemble


@torch.no_grad()
def autoregressive_rollout_metrics(
    dynamics: SystemDynamicsEnsemble,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rollout_horizon: int,
    horizons: Iterable[int] = (1, 8, 32, 100),
) -> dict[str, float]:
    states, actions, next_states, contacts, terminations = batch
    history = int(dynamics.cfg.history_horizon)
    max_rollout = min(int(rollout_horizon), int(states.shape[1] - history))
    wanted = {int(h) for h in horizons if int(h) <= max_rollout}
    if max_rollout <= 0:
        return {}

    state_hist = states[:, :history].clone()
    action_hist = actions[:, :history].clone()
    mse_by_horizon: dict[int, float] = {}
    normalized_errors: list[torch.Tensor] = []
    epistemic_values: list[torch.Tensor] = []
    aleatoric_values: list[torch.Tensor] = []
    contact_accs: list[torch.Tensor] = []
    termination_accs: list[torch.Tensor] = []

    for step in range(max_rollout):
        pred_state, aleatoric, epistemic, contact_logits, term_logits = dynamics.predict(state_hist, action_hist)
        target_idx = history - 1 + step
        target_state = next_states[:, target_idx]
        mse = (pred_state - target_state).square().mean(dim=-1)
        denom = target_state.abs().sum(dim=-1).clamp_min(1e-6)
        norm_error = (pred_state - target_state).abs().sum(dim=-1) / denom
        if (step + 1) in wanted:
            mse_by_horizon[step + 1] = float(mse.mean().detach().cpu())
        normalized_errors.append(norm_error.detach())
        epistemic_values.append(epistemic.detach())
        aleatoric_values.append(aleatoric.detach())

        contact_pred = torch.sigmoid(contact_logits) > 0.5
        contact_target = contacts[:, target_idx] > 0.5
        contact_accs.append((contact_pred == contact_target).float().mean())
        term_pred = torch.sigmoid(term_logits) > 0.5
        term_target = terminations[:, target_idx] > 0.5
        termination_accs.append((term_pred == term_target).float().mean())

        state_hist = torch.cat([state_hist[:, 1:], pred_state.unsqueeze(1)], dim=1)
        if step + 1 < max_rollout:
            next_action = actions[:, history + step]
            action_hist = torch.cat([action_hist[:, 1:], next_action.unsqueeze(1)], dim=1)

    out: dict[str, float] = {}
    for horizon, value in mse_by_horizon.items():
        out[f"{horizon}_step_mse"] = value
    errors = torch.cat(normalized_errors).detach().cpu().float()
    epistemic = torch.cat(epistemic_values).detach().cpu().float()
    aleatoric = torch.cat(aleatoric_values).detach().cpu().float()
    out["traj_autoregressive_error"] = float(errors.mean())
    out["epistemic_uncertainty_mean"] = float(epistemic.mean())
    out["aleatoric_uncertainty_mean"] = float(aleatoric.mean())
    out["contact_accuracy"] = float(torch.stack(contact_accs).mean().detach().cpu())
    out["termination_accuracy"] = float(torch.stack(termination_accs).mean().detach().cpu())

    err_np = errors.numpy()
    epi_np = epistemic.numpy()
    if err_np.size > 1 and np.std(err_np) > 1e-8 and np.std(epi_np) > 1e-8:
        out["epistemic_error_correlation"] = float(np.corrcoef(epi_np, err_np)[0, 1])
    else:
        out["epistemic_error_correlation"] = 0.0
    return out
