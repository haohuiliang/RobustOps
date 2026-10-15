"""Pre-update alignment summaries and bounded exact-token anomaly evidence."""

import json
import math
from pathlib import Path
import numpy as np
import torch
from robustops.training.recovery_rl_core import write_json
from robustops.training.rl_action_feedback import (
    flow_multipliers,
    group_signal_diagnostics,
)


def record_alignment(batch, step, cfg):
    data = batch.batch
    mask = data["response_mask"].detach().cpu()
    actor = data["old_log_probs"].detach().float().cpu()
    rollout = data["rollout_log_probs"].detach().float().cpu()
    if (
        mask.shape != actor.shape
        or mask.shape != rollout.shape
        or not torch.all((mask == 0) | (mask == 1))
    ):
        raise ValueError("alignment token/mask shape or binary-mask contract failed")
    mask = mask.bool()
    if not mask.any() or not (
        torch.isfinite(actor[mask]).all() and torch.isfinite(rollout[mask]).all()
    ):
        raise ValueError(
            "alignment has missing or nonfinite generated-token probabilities"
        )
    delta = (actor - rollout).abs().masked_fill(~mask, 0)
    probability_delta = (actor.exp() - rollout.exp()).abs().masked_fill(~mask, 0)
    tail = mask & ((delta > math.log(10)) | (probability_delta > 0.2))

    def statistics(name, values):
        values = values[mask]
        return {
            f"alignment/{name}_{key}": value
            for key, value in dict(
                mean=values.mean().item(),
                max=values.max().item(),
                p99=torch.quantile(values, 0.99).item(),
                std=values.std(correction=0).item(),
            ).items()
        }

    metrics = dict(
        statistics("abs_log_ratio", delta),
        **statistics("abs_probability_difference", probability_delta),
    )
    metrics.update(
        {
            "alignment/generated_tokens": int(mask.sum()),
            "alignment/trajectories": len(mask),
            "alignment/tail_tokens": int(tail.sum()),
            "alignment/tail_token_fraction": float(tail.sum() / mask.sum()),
            "alignment/tail_trajectories": int(tail.any(-1).sum()),
            "alignment/tail_trajectory_fraction": float(tail.any(-1).float().mean()),
        }
    )
    root = Path(cfg.output) / "alignment" / f"step_{step}"
    score = torch.maximum(delta / math.log(10), probability_delta / 0.2)
    order = score.max(-1).values.argsort(descending=True)
    evidence = []
    for index in order.tolist():
        if len(evidence) == 4 or not tail[index].any():
            break
        valid = mask[index].nonzero().flatten()
        positions = valid[score[index, valid].argsort(descending=True)[:32]].tolist()
        entry = dict(
            batch_row=index,
            token_positions=positions,
            artifact=f"row_{index}.npz",
            update_step=step,
            actor_policy_version=step - 1,
            rollout_policy_version=step - 1,
            version_basis="synchronous manager return before optimizer update",
            weights_retained_for_this_version=False,
        )
        for key in (
            "uid",
            "recovery_visit",
            "recovery_task_id",
            "recovery_mode",
            "recovery_flow",
            "recovery_parent_visit",
            "recovery_source_state",
            "recovery_episode",
            "min_global_steps",
            "max_global_steps",
        ):
            value = batch.non_tensor_batch.get(key)
            if value is not None:
                item = value[index]
                entry[key] = item.item() if isinstance(item, np.generic) else item
        root.mkdir(parents=True, exist_ok=True)
        tensors = {
            key: data[key][index].detach().cpu().numpy()
            for key in (
                "input_ids",
                "prompts",
                "responses",
                "attention_mask",
                "position_ids",
                "response_mask",
            )
        }
        np.savez_compressed(
            root / entry["artifact"],
            **tensors,
            actor_log_probs=actor[index].numpy(),
            rollout_log_probs=rollout[index].numpy(),
        )
        evidence.append(entry)
    from omegaconf import OmegaConf

    write_json(
        root / "summary.json",
        dict(
            metrics=metrics,
            evidence=evidence,
            recovery_config=OmegaConf.to_container(cfg, resolve=True)
            if OmegaConf.is_config(cfg)
            else vars(cfg),
            thresholds=dict(abs_log_ratio=math.log(10), abs_probability_difference=0.2),
            capture_limit_trajectories=4,
            capture_limit_highlight_tokens=32,
            scope="actual generated tokens, pre-update actor vs rollout",
            finite_tails_stop_training=False,
        ),
    )
    return metrics


def prepare_joint_loss(batch, cfg=None):
    clean_groups = 16
    mask = batch.batch["response_mask"]
    flows = list(batch.non_tensor_batch["recovery_flow"])
    factors, weights = flow_multipliers(flows, mask.sum(-1).tolist())
    batch.batch["recovery_loss_weights"] = (
        torch.tensor(factors, device=mask.device, dtype=torch.float32)
        .unsqueeze(-1)
        .expand_as(mask)
        .clone()
    )
    rewards = batch.batch["token_level_rewards"].sum(-1)
    metrics = {
        "recovery/batch_reward_std": rewards.std(correction=1).item(),
        "recovery/nonzero_advantage_tokens": int(
            (batch.batch["advantages"] != 0).sum()
        ),
    }
    for flow in sorted(set(flows)):
        selected = torch.tensor([s == flow for s in flows], device=mask.device)
        metrics[f"recovery/{flow}/rows"] = int(selected.sum())
        metrics[f"recovery/{flow}/generated_tokens"] = int(mask[selected].sum())
        metrics[f"recovery/{flow}/loss_weight"] = weights.get(flow, 0.0)
        if selected.any():
            metrics[f"recovery/{flow}/reward_mean"] = rewards[selected].mean().item()
    # These scalars enter TensorBoard through the trainer's usual metrics writer.
    all_metrics = list(batch.non_tensor_batch["recovery_metrics"])
    for condition in (False, True):
        name = "actual_fault" if condition else "actual_clean"
        local = [m for m in all_metrics if bool(m["actual_hit"]) == condition]
        metrics[f"recovery/{name}/total"] = len(local)
        metrics[f"recovery/{name}/successes"] = sum(m["strict_success"] for m in local)
        if local:
            metrics[f"recovery/{name}/success_pct"] = (
                100 * sum(m["strict_success"] for m in local) / len(local)
            )
    metrics["recovery/planned_fault_parent_fraction"] = 0.5
    metrics["recovery/actual_fault_row_fraction"] = sum(
        m["actual_hit"] for m in all_metrics
    ) / len(all_metrics)
    metrics["recovery/enabled_row_fraction"] = sum(
        m["fault_enabled"] for m in all_metrics
    ) / len(all_metrics)
    slots = list(batch.non_tensor_batch["recovery_joint_slot"])
    metrics["recovery/original_clean_success_pct"] = (
        100
        * sum(
            m["strict_success"]
            for i, m in enumerate(all_metrics)
            if slots[i] < clean_groups
        )
        / (4 * clean_groups)
    )
    metrics["recovery/fallback_groups"] = (
        sum(flows[i] == "clean" and slots[i] >= clean_groups for i in range(len(flows)))
        / 4
    )
    for flow, counts in group_signal_diagnostics(all_metrics, flows, slots).items():
        if flow not in flows:
            continue
        for name, value in counts.items():
            metrics[f"recovery/{flow}/{name}"] = value
    for flow in sorted(set(flows)):
        indices = [i for i, value in enumerate(flows) if value == flow]
        zero_slots = {
            slot
            for slot in set(slots[i] for i in indices)
            if not bool(
                batch.batch["advantages"][
                    [i for i in indices if slots[i] == slot]
                ].any()
            )
        }
        for key in (
            "episode_seconds",
            "generation_request_seconds",
            "generated_tokens",
            "tool_calls",
        ):
            metrics[f"recovery/{flow}/{key}_sum"] = sum(
                all_metrics[i][key] for i in indices
            )
            metrics[f"recovery/{flow}/zero_signal_{key}_sum"] = sum(
                all_metrics[i][key] for i in indices if slots[i] in zero_slots
            )
    for key in (
        "bad_action_count",
        "bad_action_invalid_arguments",
        "bad_action_unexposed_tool",
        "actual_dose",
        "new_dose",
        "inherited_dose",
        "tool_calls",
        "budget_terminated",
    ):
        metrics[f"recovery/{key}"] = sum(m[key] for m in all_metrics)
    return metrics
