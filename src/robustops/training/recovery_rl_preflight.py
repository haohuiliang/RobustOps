"""Run after the configured hardware checks, before starting owned MCP services."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    command = json.loads(Path(args.command).read_text())
    root = Path(__file__).resolve().parents[3]
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.utils.config import validate_config, omega_conf_to_dataclass
    from robustops.training.rl_agent import RecoveryAgentLoop  # noqa: F401 - verify native runtime imports
    from robustops.training.rl_manager import RecoveryManager  # noqa: F401 - verify native runtime imports
    from robustops.training.recovery_rl_bridge import TokenBridge
    from transformers import AutoTokenizer

    with initialize_config_dir(
        config_dir=str(root / "verl/verl/trainer/config"), version_base=None
    ):
        cfg = compose(config_name="ppo_trainer", overrides=command[3:])
    OmegaConf.resolve(cfg)
    from robustops.training.rl_credit import prepare_advantages, post_update_metrics

    # Tiny production tensor-path check, only after the launcher's hardware gate.
    import torch
    from types import SimpleNamespace

    rows = [
        dict(
            reward_success=float(i % 4 == 0),
            reward_progress=(i % 4) / 4,
            reward_quality=float(i % 4 == 0),
            strict_success=int(i % 4 == 0),
            reward_gain=0.0,
            reward_loss=0.0,
            reward_bad_ratio=0.0,
            generated_tokens=2,
            bad_action_count=0,
        )
        for i in range(128)
    ]
    probe = SimpleNamespace(
        batch=dict(response_mask=torch.tensor([[1, 1, 0]] * 128)),
        non_tensor_batch=dict(
            recovery_metrics=rows,
            recovery_joint_slot=[i // 4 for i in range(128)],
            recovery_bad_spans=[
                [dict(start=1, end=2, rule="repeated_no_progress_read")]
            ]
            + [[] for _ in range(127)],
            recovery_flow=["clean"] * 64 + ["fault"] * 64,
        ),
    )
    prepare_advantages(probe, cfg.recovery)
    assert probe.batch["advantages"][0, 1] <= -cfg.recovery.local_bad_eta
    assert not probe.batch["advantages"][:, 2].any()
    probe.batch["old_log_probs"] = torch.full((128, 3), -1.0)
    probe.batch["recovery_loss_weights"] = torch.ones((128, 3))
    assert (
        post_update_metrics(probe, probe.batch["old_log_probs"])[
            "reward/update/weighted_logprob_delta"
        ]
        == 0.0
    )
    after = probe.batch["old_log_probs"] + 0.01 * probe.batch["advantages"]
    assert (
        post_update_metrics(probe, after)["reward/update/weighted_logprob_delta"] > 0.0
    )
    validate_config(cfg, use_reference_policy=True, use_critic=False)
    omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.actor_rollout_ref.model.path, local_files_only=True
    )
    # Exercise real tokenizer/template concatenation without model weights or MCP.
    initial = tokenizer.apply_chat_template(
        [dict(role="user", content="preflight")],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    bridge = TokenBridge(tokenizer, None, {}, None, cfg.recovery.max_generation_tokens, 0)
    bridge.ids = initial + tokenizer.encode(
        '<tool_call>\n{"name":"read","arguments":{}}\n</tool_call>',
        add_special_tokens=False,
    )
    suffix = tokenizer.decode(
        bridge.suffix([dict(role="tool", content="tool result")]),
        skip_special_tokens=False,
    )
    if not suffix.startswith(
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
    ) or not suffix.endswith("<|im_start|>assistant\n"):
        raise ValueError("unsupported chat template for exact-token recovery bridge")
    import pyarrow.parquet as pq
    from collections import defaultdict

    cells = defaultdict(set)
    for file in (cfg.data.train_files, cfg.data.val_files):
        for item in pq.read_table(file).to_pylist():
            row = item["extra_info"]["recovery"]
            cells[(row["domain"], row["mode"])].add(row["task_id"])
    from robustops.envs.enterpriseops.task import load_tasks

    for (domain, mode), required in cells.items():
        directory = "oracle" if mode == "oracle" else "plus5"
        recipe = json.loads(
            (
                root / f"recipes/enterpriseops/paired/{directory}/{domain}.json"
            ).read_text()
        )
        tasks = load_tasks(
            root / recipe["data_path"],
            expected_domain=domain,
            expected_count=recipe["expected_task_count"],
        )
        if not required <= {task.task_id for task in tasks}:
            raise ValueError(f"unloadable task IDs: {domain}/{mode}")
        if cfg.recovery.get("verifier_graph_dir"):
            from robustops.training.recovery_verifier_graph import task_graph

            for task in tasks:
                if task.task_id in required:
                    task_graph(cfg.recovery.verifier_graph_dir, task, mode)
        del tasks
    print(
        "RECOVERY_PREFLIGHT_OK: config, native imports, tokenizer, full task identities; no GPU update yet",
        flush=True,
    )


if __name__ == "__main__":
    main()
