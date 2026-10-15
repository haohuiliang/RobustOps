"""Reward components and token-local credit for the shared GRPO objective."""

from collections import Counter
import json
import math


def components(boundary, final, success, bad_count, tool_calls, *, dependencies=None):
    if not boundary or len(boundary) != len(final):
        raise ValueError("boundary verifier coverage mismatch")
    for a, b in zip(boundary, final, strict=True):
        if (
            (a["index"], a["name"]) != (b["index"], b["name"])
            or type(a["passed"]) is not bool
            or type(b["passed"]) is not bool
        ):
            raise ValueError("boundary verifier identity/outcome mismatch")
    if not 0 <= bad_count <= tool_calls:
        raise ValueError("bad calls must be a subset of new tool attempts")
    gain = sum(not a["passed"] and b["passed"] for a, b in zip(boundary, final)) / len(
        final
    )
    loss = sum(a["passed"] and not b["passed"] for a, b in zip(boundary, final)) / len(
        final
    )
    graph_metrics = dict(
        reward_progress_raw=gain - loss,
        reward_gain_raw=gain,
        reward_loss_raw=loss,
        reward_graph_enabled=0,
    )
    if dependencies is not None:
        from robustops.data.verifier_graph import progress

        expected = list(range(1, len(final) + 1))
        if list(dependencies) != expected or [v["index"] for v in boundary] != expected:
            raise ValueError("verifier graph/result index mismatch")
        before = progress([v["passed"] for v in boundary], dependencies)
        after = progress([v["passed"] for v in final], dependencies)
        graph_metrics = dict(
            reward_progress_raw=gain - loss,
            reward_gain_raw=gain,
            reward_loss_raw=loss,
            reward_graph_boundary_score=sum(before) / len(before),
            reward_graph_final_score=sum(after) / len(after),
            reward_graph_enabled=1,
        )
        gain = sum(not a and b for a, b in zip(before, after)) / len(after)
        loss = sum(a and not b for a, b in zip(before, after)) / len(after)
    cost = bad_count / max(1, tool_calls)
    return dict(
        reward_success=float(success),
        reward_progress=gain - loss,
        reward_quality=float(success) * (1 - cost),
        reward_gain=gain,
        reward_loss=loss,
        reward_bad_ratio=cost,
        **graph_metrics,
    )


def exact_bad_spans(bridge, trajectory, evidence):
    ids = bridge.ids[len(bridge.prompt_ids) :]
    spans = []
    turns = {t["turn"]: t for t in trajectory["turns"]}
    opener, closer = [
        bridge.tokenizer.convert_tokens_to_ids(t)
        for t in ("<tool_call>", "</tool_call>")
    ]
    for event in evidence:
        usage = turns[event["turn"]]["reported_usage"]
        start = usage["prompt_tokens"] - len(bridge.prompt_ids)
        end = start + usage["completion_tokens"]
        if not 0 <= start < end <= len(ids) or not all(bridge.mask[start:end]):
            raise ValueError("bad action is outside exact newly generated tokens")
        part = ids[start:end]
        if part.count(opener) != 1 or part.count(closer) != 1:
            raise ValueError("bad action does not map to a unique tool call")
        left, right = start + part.index(opener), start + part.index(closer) + 1
        if left >= right:
            raise ValueError("invalid tool call delimiter order")
        parsed = json.loads(
            bridge.tokenizer.decode(
                ids[left + 1 : right - 1], skip_special_tokens=False
            )
        )
        if parsed["name"] != event["name"] or parsed["arguments"] != event["arguments"]:
            raise ValueError("executed action differs from sampled token span")
        spans.append(
            dict(start=left, end=right, turn=event["turn"], rule=event["rule"])
        )
    return spans


def prepare_advantages(batch, cfg):
    import torch

    mask = batch.batch["response_mask"]
    device = mask.device
    rows = list(batch.non_tensor_batch["recovery_metrics"])
    slots = list(batch.non_tensor_batch["recovery_joint_slot"])
    temporal = getattr(cfg, "q_temporal_credit", False)
    quality = [0.0] * len(rows)
    from robustops.training.rl_advantage import advantages

    scalars, metrics, quality = advantages(
        rows,
        slots,
        list(cfg.reward_weights),
        cfg.reward_normalization,
        return_quality=True,
    )
    from robustops.training.recovery_local_credit import credit_segments

    segments = [
        credit_segments(
            m,
            spans,
            a,
            q,
            float(cfg.local_bad_eta),
            temporal=temporal,
            bad_count=row.get("bad_action_count"),
            terminal_stop_cap=getattr(cfg, "terminal_stop_advantage_cap", None),
        )
        for m, spans, a, q, row in zip(
            mask, batch.non_tensor_batch["recovery_bad_spans"], scalars, quality, rows
        )
    ]
    advantage = torch.zeros_like(mask, dtype=torch.float32)
    for i, parts in enumerate(segments):
        for part in parts:
            left, right = part["start"], part["end"]
            advantage[i, left:right] = part["before"] * mask[i, left:right]
    # Exact segments also feed coefficient accounting without re-expanding tokens.
    if not hasattr(batch, "meta_info"):
        batch.meta_info = {}
    batch.meta_info["recovery_credit_segments"] = segments
    if temporal:
        metrics["normalization/pre_temporal_trajectory_advantage_rms"] = metrics[
            "reward/trajectory_advantage_rms"
        ]
        means = [
            sum(p["tokens"] * p["before"] for p in parts)
            / sum(p["tokens"] for p in parts)
            for parts in segments
        ]
        metrics["reward/trajectory_advantage_rms"] = math.sqrt(
            sum(a * a for a in means) / len(means)
        )
        metrics["q_temporal/mean"] = sum(
            p["tokens"] * p["quality"] for parts in segments for p in parts
        ) / float(mask.sum())
        metrics["q_temporal/zero_suffix_tokens"] = sum(
            p["tokens"] for parts in segments for p in parts if p["q_fraction"] == 0
        )
    bad = torch.zeros_like(mask, dtype=torch.bool)
    repeat = torch.zeros_like(bad)
    terminal = torch.zeros_like(bad)
    for i, spans in enumerate(batch.non_tensor_batch["recovery_bad_spans"]):
        for span in spans:
            left, right = span["start"], span["end"]
            if not 0 <= left < right <= mask.shape[1] or not mask[i, left:right].all():
                raise ValueError("packed bad span changed generated-token mask")
            bad[i, left:right] = True
            if span.get("rule") == "failed_native_stop":
                terminal[i, left:right] = True
            if span.get("rule") == "repeated_no_progress_read":
                repeat[i, left:right] = True
    metrics["reward/positive_bad_tokens_before"] = int((bad & (advantage > 0)).sum())
    eta = float(cfg.local_bad_eta)
    if eta < 0 or not math.isfinite(eta):
        raise ValueError("local_bad_eta must be finite and nonnegative")
    metrics.update(
        {
            "local_read_repeat/tokens": int(repeat.sum()),
            "local_read_repeat/trajectories": int(repeat.any(dim=1).sum()),
            "local_read_repeat/positive_tokens_before": int(
                (repeat & (advantage > 0)).sum()
            ),
            "local_read_repeat/coefficient_change_abs": float(
                torch.clamp(advantage[repeat] + eta, min=0).sum()
            )
            if eta
            else 0.0,
            "local_read_repeat/zero_to_nonzero_groups": sum(
                all(
                    not bool(advantage[i].any())
                    and not bool((bad[i] & ~repeat[i]).any())
                    for i, s in enumerate(slots)
                    if s == slot
                )
                and any(bool(repeat[i].any()) for i, s in enumerate(slots) if s == slot)
                for slot in set(slots)
            )
            if eta
            else 0,
        }
    )
    after_local = advantage.clone()
    for i, parts in enumerate(segments):
        for part in parts:
            after_local[i, part["start"] : part["end"]] = (
                part["after"] * mask[i, part["start"] : part["end"]]
            )
    metrics.update(
        {
            "terminal_stop/tokens": int(terminal.sum()),
            "terminal_stop/penalized_trajectories": int(terminal.any(dim=1).sum()),
            "terminal_stop/positive_tokens_before": int(
                (terminal & (advantage > 0)).sum()
            ),
            "terminal_stop/coefficient_change_abs": float(
                (after_local[terminal] - advantage[terminal]).abs().sum()
            ),
            "terminal_stop/zero_to_nonzero_groups": sum(
                all(
                    not bool(advantage[i].any())
                    and not bool((bad[i] & ~terminal[i]).any())
                    for i, s in enumerate(slots)
                    if s == slot
                )
                and any(
                    bool(after_local[i].any()) for i, s in enumerate(slots) if s == slot
                )
                for slot in set(slots)
            )
            if eta
            else 0,
        }
    )
    batch.batch["recovery_terminal_stop_mask"] = terminal
    batch.batch["recovery_local_override_mask"] = bad & (advantage != after_local)
    advantage = after_local
    batch.batch["advantages"] = advantage
    batch.batch["returns"] = advantage.clone()
    # Diagnostic mask stays on driver; the actor consumes only advantages and existing weights.
    batch.batch["recovery_bad_mask"] = bad
    metrics.update(
        {
            "reward/positive_bad_tokens_after": int((bad & (advantage > 0)).sum()),
            "reward/bad_tokens": int(bad.sum()),
            "terminal_stop/positive_tokens_after": int(
                (terminal & (advantage > 0)).sum()
            ),
            "reward/zero_pg_groups": sum(
                not bool(advantage[[i for i, s in enumerate(slots) if s == slot]].any())
                for slot in set(slots)
            ),
        }
    )
    for flow in ("clean", "fault"):
        indices = [
            i
            for i, f in enumerate(batch.non_tensor_batch["recovery_flow"])
            if f == flow
        ]
        if indices:
            metrics[f"reward/{flow}/success_pct"] = (
                100 * sum(rows[i]["strict_success"] for i in indices) / len(indices)
            )
            metrics[f"reward/{flow}/progress"] = sum(
                rows[i]["reward_progress"] for i in indices
            ) / len(indices)
    for key in ("reward_gain", "reward_loss", "reward_bad_ratio"):
        metrics[f"reward/{key[7:]}/mean"] = sum(r[key] for r in rows) / len(rows)
    return metrics


def post_update_metrics(batch, post_log_probs):
    import torch

    mask = batch.batch["response_mask"].bool()
    before = batch.batch["old_log_probs"]
    if (
        post_log_probs.shape != before.shape
        or not torch.isfinite(post_log_probs[mask]).all()
    ):
        raise ValueError("invalid post-update probe log probabilities")
    delta = post_log_probs - before
    adv = batch.batch["advantages"]
    weight = batch.batch["recovery_loss_weights"]
    bad = batch.batch["recovery_bad_mask"]
    success = torch.tensor(
        [r["strict_success"] for r in batch.non_tensor_batch["recovery_metrics"]],
        device=mask.device,
    ).bool()[:, None]
    metrics = {
        "reward/update/weighted_logprob_delta": float(
            (delta * adv * weight)[mask].mean()
        ),
        "reward/update/abs_logprob_delta": float(delta[mask].abs().mean()),
    }
    for name, selected in dict(
        success=mask & success,
        bad=mask & bad,
        terminal_stop=mask & batch.batch["recovery_terminal_stop_mask"],
        overridden_bad=mask & batch.batch["recovery_local_override_mask"],
        positive=mask & (adv > 0),
        negative=mask & (adv < 0),
    ).items():
        metrics[f"reward/update/{name}_tokens"] = int(selected.sum())
        if selected.any():
            metrics[f"reward/update/{name}_logprob_delta"] = float(
                delta[selected].mean()
            )
    for flow in ("clean", "fault"):
        selected_rows = torch.tensor(
            [f == flow for f in batch.non_tensor_batch["recovery_flow"]],
            device=mask.device,
        )
        selected = mask & selected_rows[:, None]
        if selected.any():
            metrics[f"reward/update/{flow}/weighted_logprob_delta"] = float(
                (delta * adv * weight)[selected].mean()
            )
    return metrics


def credit_accounting(rows, slots, flows, bad_tokens, cfg, *, segments=None):
    """Exact coefficient accounting, not gradient attribution; small scalar inputs only."""
    from robustops.training.rl_advantage import advantages

    scalars, _ = advantages(
        rows, slots, list(cfg.reward_weights), cfg.reward_normalization
    )
    if getattr(cfg, "q_temporal_credit", False) and segments is None:
        raise ValueError("temporal Q accounting requires exact token segments")
    flow_rows, flow_tokens = Counter(flows), Counter()
    for row, flow in zip(rows, flows):
        flow_tokens[flow] += row["generated_tokens"]
    masses = {flow: Counter() for flow in flow_rows}
    entries = []
    for i, (row, flow, a, bad) in enumerate(zip(rows, flows, scalars, bad_tokens)):
        n = row["generated_tokens"]
        factor = flow_rows[flow] / len(rows) / flow_tokens[flow]
        local = min(a, -float(cfg.local_bad_eta)) if cfg.local_bad_eta else a
        parts = (
            segments[i]
            if segments is not None
            else [
                dict(tokens=n - bad, before=a, after=a, bad=False, q_fraction=1.0),
                dict(tokens=bad, before=a, after=local, bad=True, q_fraction=1.0),
            ]
        )
        if (
            sum(p["tokens"] for p in parts) != n
            or sum(p["tokens"] for p in parts if p["bad"]) != bad
        ):
            raise ValueError("credit segments differ from actor token counts")
        values = dict(
            original_abs=sum(p["tokens"] * abs(p["before"]) for p in parts) * factor,
            final_abs=sum(p["tokens"] * abs(p["after"]) for p in parts) * factor,
            local_change=sum(p["tokens"] * abs(p["after"] - p["before"]) for p in parts)
            * factor,
            bad_negative=sum(
                p["tokens"] * max(0, -p["after"]) for p in parts if p["bad"]
            )
            * factor,
            positive_unlabelled=sum(
                p["tokens"] * max(0, p["before"]) for p in parts if not p["bad"]
            )
            * factor,
        )
        masses[flow].update(values)
        entries.append(
            dict(
                row=i,
                slot=int(slots[i]),
                flow=flow,
                success=row["strict_success"],
                S=row["reward_success"],
                P=row["reward_progress"],
                Q=row["reward_quality"],
                trajectory_advantage=a,
                trajectory_mean_before_local=sum(
                    p["tokens"] * p["before"] for p in parts
                )
                / n,
                bad_advantage=sum(p["tokens"] * p["after"] for p in parts if p["bad"])
                / bad
                if bad
                else local,
                bad_tokens=bad,
                generated_tokens=n,
                segments=parts,
                final_square_sum=sum(p["tokens"] * p["after"] ** 2 for p in parts),
                token_loss_factor=factor,
                coefficient_mass=values,
            )
        )
    total_abs = sum(v["final_abs"] for v in masses.values())
    original_abs = sum(v["original_abs"] for v in masses.values())
    metrics = {}
    for flow, counts in masses.items():
        for key, value in counts.items():
            metrics[f"reward/credit/{flow}/{key}"] = value
        metrics[f"reward/credit/{flow}/abs_share"] = (
            counts["final_abs"] / total_abs if total_abs else 0.0
        )
    metrics["reward/credit/local_change_ratio"] = (
        sum(v["local_change"] for v in masses.values()) / original_abs
        if original_abs
        else 0.0
    )
    scalars = [e["trajectory_mean_before_local"] for e in entries]
    groups = [
        [i for i, s in enumerate(slots) if s == slot] for slot in sorted(set(slots))
    ]
    mixed = [
        idx for idx in groups if 0 < sum(rows[i]["strict_success"] for i in idx) < 4
    ]
    successful = [scalars[i] for idx in mixed for i in idx if rows[i]["strict_success"]]
    failed = [
        scalars[i]
        for idx in groups
        if not any(rows[i]["strict_success"] for i in idx)
        for i in idx
        if scalars[i] > 0
    ]
    metrics["reward/credit/mixed_success_nonpositive"] = sum(a <= 0 for a in successful)
    metrics["reward/credit/mixed_failure_nonnegative"] = sum(
        scalars[i] >= 0 for idx in mixed for i in idx if not rows[i]["strict_success"]
    )
    if successful and failed and min(successful) > 0:
        metrics["reward/credit/max_failed_to_min_mixed_success"] = max(failed) / min(
            successful
        )
    weighted_success = [
        scalars[i] * entries[i]["token_loss_factor"]
        for idx in mixed
        for i in idx
        if rows[i]["strict_success"]
    ]
    weighted_failure = [
        scalars[i] * entries[i]["token_loss_factor"]
        for idx in groups
        if not any(rows[i]["strict_success"] for i in idx)
        for i in idx
        if scalars[i] > 0
    ]
    if weighted_success and weighted_failure and min(weighted_success) > 0:
        metrics["reward/credit/weighted_max_failed_to_min_mixed_success"] = max(
            weighted_failure
        ) / min(weighted_success)
    return metrics, entries


def flow_signal_summary(rows, slots, flows, parents, zero_slots, original_groups=32):
    """Separate reward spread from actual PG signal after local supervision."""
    metrics = {}
    for flow in sorted(set(flows)):
        indices = [i for i, f in enumerate(flows) if f == flow]
        flow_slots = set(slots[i] for i in indices)
        metrics[f"reward/{flow}/strict_mixed_groups"] = sum(
            0 < sum(rows[i]["strict_success"] for i in indices if slots[i] == slot) < 4
            for slot in flow_slots
        )
        metrics[f"reward/{flow}/zero_pg_groups"] = len(flow_slots & zero_slots)
        metrics[f"reward/{flow}/zero_pg_generated_tokens"] = sum(
            rows[i]["generated_tokens"] for i in indices if slots[i] in zero_slots
        )
        metrics[f"reward/{flow}/zero_pg_episode_seconds"] = sum(
            rows[i]["episode_seconds"] for i in indices if slots[i] in zero_slots
        )
        metrics[f"reward/{flow}/all_failed_groups"] = sum(
            not any(rows[i]["strict_success"] for i in indices if slots[i] == slot)
            for slot in flow_slots
        )
    original_parents = {
        parents[i] for i in range(len(rows)) if slots[i] < original_groups
    }
    metrics["reward/original_parents/any_success"] = sum(
        any(rows[i]["strict_success"] for i in range(len(rows)) if parents[i] == parent)
        for parent in original_parents
    )
    metrics["reward/original_parents/no_success"] = (
        len(original_parents) - metrics["reward/original_parents/any_success"]
    )
    return metrics
