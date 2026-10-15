"""GRPO advantages with aggregate or per-component reward normalization."""

from collections import Counter
from statistics import mean, stdev
import math

CLEAN, FAULT, GROUPS = 16, 16, 32
NORMALIZATION_METHODS = ("aggregate", "component")


def standardized(values):
    center, scale = mean(values), stdev(values) + 1e-6
    return [
        0.0 if abs((v - center) / scale) < 1e-12 else (v - center) / scale
        for v in values
    ], scale


def advantages(rows, slots, weights, style, *, group_count=32, return_quality=False):
    if style not in NORMALIZATION_METHODS:
        raise ValueError("unknown normalization style")
    if (
        group_count not in (1, 32)
        or len(rows) != group_count * 4
        or len(set(slots)) != group_count
        or any(slots.count(s) != 4 for s in set(slots))
    ):
        raise ValueError("reward normalization requires complete G4 groups")
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("reward_weights must contain three finite nonnegative values")
    keys = ("reward_success", "reward_progress", "reward_quality")
    raw = [[r[k] for k in keys] for r in rows]
    if any(not math.isfinite(v) for row in raw for v in row):
        raise ValueError("nonfinite reward component")
    combined, contributions, categories = (
        [0.0] * len(rows),
        [[0.0] * len(weights) for _ in rows],
        {},
    )
    groups = [
        [i for i, s in enumerate(slots) if s == slot] for slot in sorted(set(slots))
    ]
    for ids in groups:
        values = [[raw[i][j] for i in ids] for j in range(len(weights))]
        if style == "aggregate":
            total = [sum(w * v for w, v in zip(weights, raw[i])) for i in ids]
            normalized, scale = standardized(total)
            for n, i in enumerate(ids):
                combined[i] = normalized[n]
                contributions[i] = [
                    weights[j] * (raw[i][j] - mean(values[j])) / scale
                    for j in range(len(weights))
                ]
        else:
            z = [standardized(v)[0] for v in values]
            for n, i in enumerate(ids):
                contributions[i] = [weights[j] * z[j][n] for j in range(len(weights))]
                combined[i] = sum(contributions[i])
        successes = sum(raw[i][0] for i in ids)
        category = (
            "mixed_success"
            if 0 < successes < 4
            else "failed_progress"
            if successes == 0 and len(set(values[1])) > 1
            else "success_quality"
            if successes == 4 and len(set(values[2])) > 1
            else "other"
        )
        categories.setdefault(category, []).extend(ids)
    result, denominator = standardized(combined)
    metrics = {
        "reward/batch_normalizer": denominator,
        "reward/trajectory_advantage_rms": math.sqrt(mean(x * x for x in result)),
    }
    for j, key in enumerate(("success", "progress", "quality")):
        metrics[f"reward/{key}/mean"] = mean(row[j] for row in raw)
        metrics[f"reward/{key}/weighted_rms"] = math.sqrt(
            mean((r[j] / denominator) ** 2 for r in contributions)
        )
        metrics[f"reward/{key}/nonzero_groups"] = sum(
            len({raw[i][j] for i in ids}) > 1 for ids in groups
        )
    for cat, ids in categories.items():
        metrics[f"normalization/{cat}/groups"] = len(ids) // 4
        metrics[f"normalization/{cat}/advantage_rms"] = math.sqrt(
            mean(result[i] ** 2 for i in ids)
        )
    if return_quality:
        return result, metrics, [r[2] / denominator for r in contributions]
    return result, metrics


def zero_reward_advantage_slots(outputs, weights, style, *, group_count=32):
    """Refill groups without reward-component contrasts, before token-local credit overrides."""
    rows = [o.extra_fields["recovery_metrics"] for o in outputs]
    slots = [o.extra_fields["recovery_joint_slot"] for o in outputs]
    scalars, _ = advantages(rows, slots, weights, style, group_count=group_count)
    nonzero = {slot for slot, value in zip(slots, scalars) if value != 0}
    return sorted(set(slots) - nonzero)


def all_success_slots(outputs):
    groups = {}
    for out in outputs:
        groups.setdefault(out.extra_fields["recovery_joint_slot"], []).append(out)
    return sorted(
        slot
        for slot, members in groups.items()
        if len(members) == 4
        and all(o.extra_fields["recovery_metrics"]["strict_success"] for o in members)
    )


def training_filter_slots(
    outputs, weights, style, *, exclude_all_success_groups=False, **kwargs
):
    """Exclude solved G4s as well as groups without base reward contrasts."""
    zeros = zero_reward_advantage_slots(outputs, weights, style, **kwargs)
    return (
        sorted(set(zeros) | set(all_success_slots(outputs)))
        if exclude_all_success_groups
        else zeros
    )


def zero_pg_slots(
    outputs,
    weights,
    style,
    eta,
    *,
    group_count=32,
    ignore_rules=(),
    q_temporal_credit=False,
    terminal_stop_cap=None,
):
    """Match final actor PG support, including exact bad-token overrides; no KL."""
    rows = [o.extra_fields["recovery_metrics"] for o in outputs]
    slots = [o.extra_fields["recovery_joint_slot"] for o in outputs]
    scalars, _, quality = advantages(
        rows, slots, weights, style, group_count=group_count, return_quality=True
    )
    nonzero = set()
    from robustops.training.recovery_local_credit import credit_segments

    for out, slot, row, scalar, q in zip(outputs, slots, rows, scalars, quality):
        spans = [
            s
            for s in out.extra_fields["recovery_bad_spans"]
            if s.get("rule") not in ignore_rules
        ]
        segments = credit_segments(
            out.response_mask,
            spans,
            scalar,
            q,
            eta,
            temporal=q_temporal_credit,
            bad_count=row.get("bad_action_count"),
            terminal_stop_cap=terminal_stop_cap,
        )
        if any(s["after"] != 0 for s in segments):
            nonzero.add(slot)
    return sorted(set(slots) - nonzero)


def fault_exposure_summary(rows):
    """Exposure by mechanism and planned dose; members and G4 groups stay distinct."""
    strata = {}
    for row in rows:
        if row["flow"] != "fault" or not row.get("kind"):
            continue
        m = row["metrics"]
        key = f"{row['kind']}/dose_{int(m['planned_dose'])}"
        item = strata.setdefault(
            key,
            dict(
                members=0,
                hit_members=0,
                budget_met_members=0,
                actual_dose_histogram=Counter(),
                first_call_fraction_sum=0.0,
                eligible_rank_sum=0.0,
                eligible_count_sum=0.0,
                eligible_window_count_sum=0.0,
                donor_later_opportunities_sum=0.0,
                measured_position_members=0,
                budget_terminated_members=0,
            ),
        )
        actual = int(m["actual_dose"])
        item["members"] += 1
        item["hit_members"] += int(actual > 0)
        item["budget_met_members"] += int(actual >= m["planned_dose"])
        item["actual_dose_histogram"][str(actual)] += 1
        item["budget_terminated_members"] += int(m.get("budget_terminated", 0))
        if "fault_first_call_fraction" in m:
            item["measured_position_members"] += 1
            for target, source in [
                ("first_call_fraction_sum", "fault_first_call_fraction"),
                ("eligible_rank_sum", "fault_eligible_rank"),
                ("eligible_count_sum", "fault_eligible_count"),
                ("eligible_window_count_sum", "fault_eligible_window_count"),
                ("donor_later_opportunities_sum", "fault_donor_later_opportunities"),
            ]:
                item[target] += m[source]
    for item in strata.values():
        item["groups"] = item["members"] // 4
        item["budget_met_fraction"] = item["budget_met_members"] / item["members"]
        item["actual_dose_histogram"] = dict(item["actual_dose_histogram"])
    return strata
