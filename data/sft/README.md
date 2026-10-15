# SFT data

[English](README.md) · [简体中文](README.zh-CN.md)

These ready-to-use records are included in the GitHub repository. The training recipes use them directly; no Hugging Face dataset download is required.

| Split | Records | Files and role |
|---|---:|---|
| Train | 692 | `train/records.jsonl`, `train/train.parquet`, `train/provenance.jsonl` and `train/fault_exposure.json`; supervised training |
| Validation | 15 | `validation/records.jsonl`, `validation/validation.parquet` and `validation/provenance.jsonl`; a small set for quick monitoring of Val performance, without gradient updates |

These records were collected from 80% of runnable Train tasks, selected by task ID; self-collection can choose another Train manifest or fraction. Accepted demonstration coverage is measured separately against the full **1,025 runnable Train task–tool-mode pairs**:

| Condition | Records / distinct pairs | Demonstration coverage |
|---|---:|---:|
| Clean | 357 / 357 | **357/1,025 (34.83%)** |
| Fault | 335 / 335 | **335/1,025 (32.68%)** |

There are 289 pairs with both conditions, giving a deduplicated union of **403/1,025 pairs (39.32%)**.

JSONL and Parquet represent the same examples, including task identity, messages, tools and supervision masks. Use `DecisionSFTDataset` through the supplied recipes to preserve selective decision supervision and trajectory weighting. Identified invalid decisions remain visible as context with their target loss masked. A legitimate call encountering a transient fault is not automatically masked; valid encounters and subsequent recovery remain learnable.

```python
from datasets import load_dataset

records = load_dataset("parquet", data_files={
    "train": "data/sft/train/train.parquet",
    "validation": "data/sft/validation/validation.parquet",
})
```

Teacher provenance is linked by `record_id`. The source references in `provenance.jsonl` point to guided teacher attempts in the optional [trajectory archive](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories). Processed training records, supervision policies and provenance are included in this repository. See [teacher collection](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-teacher-data-collection) for scope and successful conversion.

See the [data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-collect-sft-demonstrations) for self-collection. The unified [trajectory-to-SFT module](../../src/robustops/data/teacher_sft.py) converts guided inference artifacts into complete conversations, supervision masks, mask-change reasons and Parquet; see the [processing CLI and outputs](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-convert-trajectories-to-sft-data). Val uses the same module, with a small selected subset for quick monitoring of Val performance; see [Val data processing](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-small-val-monitoring-set).

Collection budgets the student-visible context after removing teacher guidance; export retains executed history and checks the full sample against a 32K limit. See the [shared execution protocol](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-shared-execution-protocol).

## Teacher export loss masks and target behaviors

Export checks task success, valid execution, successful environment cleanup and the student’s 32K context limit before assigning supervision to assistant decisions. Extra teacher guidance is removed; executed calls, arguments, tool receipts and assistant history remain. A loss mask gives selected tokens zero supervision loss; it does not make them negative training targets.

| Behavior or content | Evidence and treatment | Learning target to preserve |
|---|---|---|
| An unexposed tool or arguments violating the exposed schema, with a rejected call | Automatically check the tool name, argument constraints and error receipt; mark the whole assistant decision `masked`, including a first occurrence | Calls that follow tool definitions and subsequent grounded corrections |
| A retained action with reviewed defects in its narration | Supplied `reviewed_annotations` select `tool_call_only`, for example when narration claims an operation that was not executed; the annotation must match the original assistant message | The actual tool action without imitating defective narration |
| Explanation or planning before the call in a `tool_call_only` decision | Retain it as input context with zero target loss | The context needed to choose the action, with supervision on the tool call |
| A legitimate call encounters a transient error, injected fault or unexpected business result | Do not automatically mask an error response; the automatic rule requires both a tool/schema violation and rejection | Reasonable attempts, use of error feedback and subsequent recovery |
| Other retained valid assistant decisions | Use `full`, including grounded narration, tool actions and completion responses | Effective execution and supported completion |
| System messages, user messages and tool responses | Always context, never assistant supervision targets | Selecting subsequent actions from the task, policy and receipts |

The three targets have precise spans: `full` covers the assistant decision’s generated tokens; `masked` supervises none of that decision; `tool_call_only` runs from `<tool_call>` to the end of the assistant message, excluding preceding prose. Planning is masked only where explicitly assigned to context, rather than across all assistant decisions.

Automatic export does not perform a complete semantic review, and eventual task success does not establish that every statement is correct. Narration masks depend on supplied reviewed annotations; the default `reviewed_annotations: null` does not generate them. For example, a rejected call using a value outside an exposed enum triggers an automatic mask, while valid arguments followed by temporary service unavailability do not. A claim that “both objects were updated” with only one executed call needs a reviewed annotation to mask the narration while retaining a learnable call.

Each trajectory’s supervision weight is divided equally among assistant decisions that are not fully masked, then equally among each decision’s supervised tokens. With total weight 1 and two decisions containing 20 and 100 supervised tokens, each decision receives 0.5; token weights are 0.025 and 0.005 respectively. A longer decision therefore receives no greater total weight.

Inspect `assistant_supervision` in exported `records.jsonl`, token spans in `masks.jsonl`, and rules and reasons in `mask_changes.jsonl`. The supplied records carry their own supervision policies.

Implementation: [export and annotations](../../src/robustops/data/teacher_sft.py) · [supervision spans and weights](../../src/robustops/training/selective_supervision.py) · [processing commands and outputs](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-convert-trajectories-to-sft-data). See [local credit rules](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/training/README.md#user-content-target-behaviors-for-local-credit) for RL; negative advantages and SFT zero-loss masks are defined separately.
