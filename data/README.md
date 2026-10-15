# Data assets

[English](README.md) · [简体中文](README.zh-CN.md)

These assets are included in the repository. Default recipes use the three supplied model-assisted products directly; you can also generate or collect your own.

| Directory | Contents |
|---|---|
| [fault_selector/](fault_selector/) | Per-tool fault labels from a Train/Val/Test selector scan: `train.jsonl`, `val.jsonl`, `test.jsonl` |
| [verifier_graph/](verifier_graph/) | Teacher-generated dependencies between Train/Val checks for RL progress feedback |
| [sft/](sft/) | 692 Train teacher demonstrations and a small set of 15 records for quick monitoring of Val performance, in JSONL/Parquet |
| [task_sets/](task_sets/) | Fixed Train/Val/Test tasks, the default 80% collection manifest and the historical SFT/RL pool manifest |

`fault_selector/` contains the dataset scan from the fault selector: potentially applicable labels per tool in each task–tool-mode pair. The LLM-generated labels narrow the runtime injector’s check scope to candidate tool–fault combinations. The injector checks eligibility using the actual calls, responses and state from clean rollouts. Its 130 Test tool records match the [fixed Test selector record](../artifacts/test_fault_policy/policy.json). Evaluation also reads [Base-fixed mechanisms and budgets](../artifacts/base_test_fixed_mechanisms.json); these serve different roles. [Full injection workflow](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-fault-injection-workflow)

The Train file retains 1,027 labels from its collection batch; current collection and training select the 1,025 runnable pairs in the effective task manifest. Val/Test contain 127/130 records. Including Test selector results does not put Test into training.

Verifier graphs provide partial-progress feedback for RL; full success still uses task-state checks. SFT keeps executed context while masking identified invalid decisions from target loss. Legitimate fault encounters and effective recovery remain learnable. The entire `ignored_filter` family is held out of project SFT/RL to evaluate generalization to an unseen fault mechanism; see [training fault records](sft/train/fault_exposure.json).

The effective population is 649 public Oracle tasks + 2 executable upstream revised supplements − 3 tasks excluded for task-definition issues = **648 tasks**: Train 518, Val 64 and Test 66. [Population metadata](task_sets/enterpriseops.json) records the excluded IDs and reasons outside all partitions. The raw `SOURCE_AUDIT.json` retains the earlier source audit; `task_sets/enterpriseops.{json,jsonl}` defines the current population.

## Generate or collect your own

Complete root README steps 1–3 and [configure the teacher API](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.md#user-content-teacher-api-configuration) using either `.env.deepseek` or the `DEEPSEEK_API_KEY` environment variable; the environment variable takes precedence. Run the commands from the repository root. Outputs go to `runs/`, leaving supplied data intact.

### Generate fault labels

```bash
# Scan Train/Val/Test for potentially applicable fault labels per tool.
bash run.sh inference --config recipes/inference/fault_selector.json
```

Complete results go to `runs/fault_selector/dataset/{train,val,test}.jsonl`. Rerun the same command to fill failed requests or structurally invalid items. Valid existing results are reused; downstream task scores do not drive relabeling.

To start an independent new scan, copy the split and export recipes:

```bash
mkdir -p .local/fault_selector
cp recipes/data/fault_selector/{all,train,val,test}.json .local/fault_selector/
```

Set each split copy's `output` to `runs/my_fault_selector/train`, `runs/my_fault_selector/val` or `runs/my_fault_selector/test`. In the `all.json` copy, point `splits` to those copies and set `output`, `staging` and `backup` to `runs/my_fault_selector/dataset`, `runs/my_fault_selector/staging` and `runs/my_fault_selector/previous_dataset`, respectively. Then run:

```bash
python -m robustops.data.fault_dataset --config .local/fault_selector/all.json
```

New labels are in `runs/my_fault_selector/dataset/{train,val,test}.jsonl`.

To collect teacher data with new labels, point `selector` in a local copy of `recipes/data/teacher.json` to the new `train.jsonl`. Val/Test evaluation first needs the labels converted to a selector `policy.json`, followed by a Base run to assign and freeze mechanisms and repetition limits. A JSONL cannot be used directly as `selector_policy`. See [use new labels for evaluation](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.md#user-content-use-new-labels-for-evaluation) for conversion and bindings.

### Generate verifier graphs

```bash
# Generate Train/Val verifier graphs.
python scripts/build_verifier_graphs.py --config recipes/data/verifier_graph.json \
  --run-dir runs/my_verifier_graph --output runs/my_verifier_graph_dataset
```

This new batch writes to `runs/my_verifier_graph_dataset/`; omitting the overrides uses the recipe defaults. Set `training.verifier_graph_dir` in a local RL recipe to use them. Graph construction is independent of fault mechanism and injection-point selection.

### Collect SFT demonstrations

```bash
# Collect, validate and export SFT demonstrations from the selected Train manifest.
python -m robustops.data.teacher_collection \
  --config recipes/data/teacher.json --stage all
```

Collection first completes every temperature-0 clean trajectory, saving state and dialogue before eligible calls. It then balances mechanisms across those observed opportunities, draws the selected repetition limit from the [strategy shared by training and inference](../recipes/enterpriseops/faults/strategy.json), and injects while continuing directly from the clean checkpoint. Collection uses runnable Train tasks and excludes `ignored_filter`; cases without eligible opportunities retain zero injections.

Cases that fail the task receive one temperature-0.6 retry.

Collection automatically calls the unified [trajectory-to-SFT module](../src/robustops/data/teacher_sft.py) to remove guidance, retain executed history and produce supervision masks and JSONL/Parquet. Identified invalid decisions are excluded from targets; legitimate transient-fault encounters and useful recovery remain learnable. Processing existing inference artifacts and applying reviewed annotations are described below.

For a new collection batch, copy `recipes/data/teacher.json` to `.local/teacher.json`, set `output` and `export` to fresh directories, and use that copy with `--config` above. The next section gives a complete example with a different Train collection scope.

Raw records go to `runs/teacher_collection/`; trajectories passing task-success and student-context checks are exported to `runs/teacher_dataset/`. Counts may differ from 692. Update `data.train_parquet`, `data.expected_train_records` and the corresponding step count in a local SFT recipe. Keep modified recipes in `.local/`; collection scope is described below.

### Choose a teacher collection manifest

The default [sft_collection_pool.json](task_sets/sft_collection_pool.json) selects the 80% Train scope used for the supplied data. Full collection accepts another compatible manifest through the local recipe's `pool` or `--pool`. For example, [recovery_train_pool.json](task_sets/recovery_train_pool.json) selects its rows marked `sft`. Teacher cases use `teacher.pool` in a local case recipe.

Generate a manifest at your own fraction:

```bash
# Example: select 60% of Train task IDs; choose your own fraction.
python -m robustops.data.collection_pool \
  --fraction 0.6 --seed 20260916 --output .local/sft_pool.json
```

The generator samples only runnable Train tasks from the current task manifest, balancing domains and tool modes. `--manifest` accepts another task JSONL with the same format. Fractions are in `(0, 1]`; existing splits and supplied manifests remain unchanged. A new manifest is sampled independently; use the supplied 80% manifest to reproduce the project's collection scope.

Copy `recipes/data/teacher.json` to `.local/teacher.json` and set `output` and `export` to new directories, then select the collection manifest:

```bash
# Alternatively, point --pool to another existing Train collection manifest.
python -m robustops.data.teacher_collection \
  --config .local/teacher.json --pool .local/sft_pool.json --stage all
```

To initialize RL from an SFT model trained on self-collected data, set `model.path` in a local RL recipe to the exported Hugging Face model directory. Use the `data` fields to bind the corresponding task pool, candidate labels, SFT records and fault-exposure sources. The launcher checks model files and data consistency without requiring exactly 692 demonstrations or a step-174 checkpoint. See the [training guide](../recipes/training/README.md) for the configuration fields.

### Convert trajectories to SFT data

[teacher_sft](../src/robustops/data/teacher_sft.py) reads `config.json`, `attempt.json` and `calls/*.request.json` from guided DeepSeek inference directories, then removes guidance, applies supervision masks, checks context and exports training data. `source` can point to one inference run or a collection containing temperature-0/0.6 rounds; the first accepted trajectory for each condition takes precedence.

```bash
# Set source, output and manifest in the copy to process existing inference artifacts.
mkdir -p .local
cp recipes/data/teacher_sft.json .local/teacher_sft.json
python -m robustops.data.teacher_sft --config .local/teacher_sft.json
```

Use a fresh `output` directory. The default recipe writes to `runs/teacher_sft/train/`:

| File | Contents |
|---|---|
| `records.jsonl` | Complete conversations without teacher guidance, tools and per-assistant supervision policies |
| `masks.jsonl` | Per-record supervision policies, token intervals receiving loss under the Base tokenizer and context checks |
| `mask_changes.jsonl` | Decisions masked or adjusted, with rules and reasons |
| `train.parquet` | SFT-ready conversations, tools and supervision policies |
| `provenance.jsonl` | Links from records to raw inference directories, conditions and temperatures |

Automatic masking covers rejected calls that violate exposed schemas. Set `reviewed_annotations` to reviewed narration annotations to exclude unsuitable explanations while retaining calls and recovery history. Training reads supervision policies from Parquet, rebuilds the loss mask with the same tokenizer and divides each trajectory's supervision weight equally among retained assistant decisions, then equally among each decision's supervised tokens. See [SFT loss masks](https://www.github.com/haohuiliang/RobustOps/blob/main/data/sft/README.md#user-content-teacher-export-loss-masks-and-target-behaviors) for behavior rules, the three supervision targets and retained content.

### Small Val monitoring set

Val uses the same module: in a recipe copy, set `partition` to `validation`, choose Val tasks through `manifest`, point `source` to their guided inference directory and set a fresh `output`. Run the processing command above. After exporting `validation/validation.parquet`, select a small subset for quick monitoring of Val performance, bind it to `data.validation_parquet` in a local SFT recipe and update `data.expected_validation_records`.

## Collect SFT Val Pass@4 trajectories

[`sft_val_pass4.json`](../recipes/inference/sft_val_pass4.json) samples each clean/fault condition four times over 127 cases, producing 1,016 trajectories at temperature 0.6. Each sample executes a full trajectory from the initial state with an independent seed. All fault samples reuse the mechanism and repetition limit in the same Base-fixed manifest. Pass@4 counts cases with at least one successful sample.

```bash
# Run SFT Val Pass@4.
bash run.sh inference --config recipes/inference/sft_val_pass4.json
# Inspect the plan on CPU without starting models or MCP.
bash run.sh inference --config recipes/inference/sft_val_pass4.json --gpus 1 --dry-run
```

The model is read from `models/sft`; output defaults to `runs/sft_val_pass4/`. Override `--gpus`, `--output` and `--base-fixed-mechanisms` as for fixed evaluation. With the default `resume: true`, rerunning the same command completes only remaining samples. Root `summary.json` contains results; `sft/summary.json` contains all four samples for each case.

These Val trajectories evaluate task completion under sampling and do not enter SFT or RL training.

[SFT format](sft/README.md) · [Optional trajectory evidence](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories) · [Upstream attribution](../THIRD_PARTY_NOTICES.md)
