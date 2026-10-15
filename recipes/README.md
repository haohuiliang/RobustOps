# Configuration entry points

[English](README.md) · [简体中文](README.zh-CN.md)

The root [README](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-1-install-the-environment) explains the reproduction flow. Configurations are grouped by operation:

| Operation | Configuration | Purpose |
|---|---|---|
| Train | `training/sft.json` | SFT on 692 records; adopt epoch 2 / step 174 |
| Train | `training/rl.json` | RL on Train tasks through step 120 |
| Inference | `inference/sft_test.json`, `inference/sft_val.json` | Fixed SFT evaluation |
| Inference | `inference/sft_val_pass4.json` | SFT Val at temperature 0.6; four independent samples per clean/fault condition |
| Inference | `inference/rl_test.json`, `inference/rl_val.json` | Fixed RL evaluation |
| Inference | `inference/base_test.json`, `inference/base_val.json` | Base controls |
| Inference | `inference/teacher_test.json`, `inference/teacher_val.json` | Unguided teacher API evaluation |
| Inference | `inference/*cases.json` | Local/API cases in evaluation or teacher-data mode |
| Data | `inference/fault_selector.json` → `data/fault_selector/all.json` | Generate preliminary tool-fault candidates |
| Data | `data/verifier_graph.json` | Generate Train/Val verifier graphs |
| Data | `data/teacher.json` | Collect teacher supervision; select a collection manifest with `pool` or `--pool` |
| Data | `data/teacher_sft.json` | Guided inference → guidance removal, masks and SFT data; [CLI and outputs](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-convert-trajectories-to-sft-data) |
| Data | `data/freeze_test.json`, `data/freeze_val.json` | Freeze faults from your own Base run |
| Data | `data/summarize.json`, `data/summarize_teacher.json` | Three-model or four-model summaries |

```bash
# Train SFT from Base using the supplied teacher data.
bash run.sh train --config recipes/training/sft.json
# Evaluate SFT on fixed Test.
bash run.sh inference --config recipes/inference/sft_test.json
```

See the [training guide](training/README.md) for GPU, checkpoint, resume and retrained-model evaluation instructions.

## Teacher API configuration

Configure the teacher API key in either of two ways:

- File: fill in `DEEPSEEK_API_KEY` in the supplied [`.env.deepseek`](../.env.deepseek).
- Environment variable: run `export DEEPSEEK_API_KEY='your-key'` in the current terminal.

Teacher Val/Test, API cases and generation or collection of all three teacher products support both methods. The environment variable takes precedence when both are configured. For individual local/API tasks, see [prompts and cases](../prompts/README.md).

## Fault injection strategy

RL, teacher collection and inference share [strategy.json](enterpriseops/faults/strategy.json). Selector labels define which tools and mechanisms to check; injectors determine eligibility from actual call arguments, responses and business state. A candidate may be ineligible for a particular call, and a trajectory may contain no eligible opportunity.

**Branch from a clean checkpoint.** New Base Val/Test assignments, first-round teacher collection and RL training use this path. During clean execution, the injector checks call eligibility and saves business state and dialogue before eligible calls. The scheduler then selects a mechanism, repetition cap and position, restores business state and dialogue, and executes that call to begin a fault continuation. Clean responses after the checkpoint are not used by the fault branch.

- Base assignment and first-round teacher collection finish all same-temperature clean cases before selection in case order, balancing fault families, mechanisms and domains with seeded tie-breaking. They use the selected mechanism’s earliest eligible position. Cases without opportunities retain zero injections.
- RL collects clean trajectories under the current policy. Each completed trajectory can enter fault-group scheduling without waiting for the entire clean batch. A position must leave enough context and turns for a fault continuation; trajectories without eligible checkpoints cannot supply a valid fault group.

**Evaluate with a fixed manifest.** Evaluation with a fixed fault configuration starts from the task’s initial state with a prescribed mechanism and cap. It injects at the first eligible call for that mechanism, then at later eligible calls until the cap is reached or the trajectory ends. A trajectory with no eligible calls has zero injections. Targeted teacher supplementation with `eligible_now` also checks online, but selects a mechanism by priority among candidates eligible at the current call and draws its cap when selected, rather than prescribing a single mechanism in advance.

**Repetition caps.** After dynamic mechanism selection, draw once per plan: transient-unexecuted uses 1/2/4 with probability 1/3 each; unknown commits and ignored filters use 1/2 with probability 1/2 each; partial reads, truncated reads and reply mixups use 1/4 with probability 1/2 each. The draw is independent of business success, and small samples need not match those proportions. Later eligible calls consume the remaining allowance; the cap does not guarantee that many actual hits.

**RL injection positions.** The default `dose_window_uniform` policy uses the eligible-position count for the selected mechanism. With `N` positions that pass the continuation-budget checks and a cap of `B`, choose uniformly among the earliest `ceil(N / B)` positions, reproducibly under a fixed seed. Cap 1 allows all positions, cap 2 the first half, and cap 4 the first quarter, rounding up to retain at least one. For `N = 10`, these windows contain 10, 5 and 3 positions. Window size counts eligible calls; token length and maximum turns filter out positions without continuation room.

A new Base run records its selected mechanism and cap for freezing and reuse by all models. Supplied fixed manifests retain their historical values; later evaluation does not redraw. The entire `ignored_filter` family is held out of project SFT/RL to evaluate generalization to an unseen fault mechanism. For teacher collection, see [teacher collection](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-collect-sft-demonstrations).

New-seed zero-hit supplementation applies only to valid, protocol-correct cases with zero actual hits under a **fixed fault configuration**. Business failure or not exhausting the cap does not trigger it. Infrastructure and protocol failures have a separate retry budget. Dynamic assignment and teacher checkpoint branching do not use this seed-changing supplementation.

See [shared strategy and assignment](../src/robustops/evaluation/fault_schedule.py), [checkpoint branching](../src/robustops/evaluation/clean_fault_branch.py), [online fixed mechanisms](../src/robustops/evaluation/selected_fault.py) and [RL position windows](../src/robustops/training/rl_fault_runtime.py).

## Generate and use a new fault assignment

Default evaluation reuses the supplied fixed manifests. The entry point below reuses selector labels, finishes this Base run’s clean cases, chooses mechanisms and caps from eligible opportunities, then branches directly and saves the assignment. Freezing reads that run’s assignment. Cases without eligible opportunities produce an explicit error listing them rather than a fabricated fixed mechanism.

```bash
# Test: run Base with the scheduler selecting faults during this run.
bash run.sh inference --config recipes/inference/base_fault_assignment_test.json \
  --output runs/my_base_test
# Freeze the mechanisms and event limits from this Base run.
python -m robustops.evaluation.reproduction \
  --action freeze --config recipes/data/freeze_test.json \
  --base-run runs/my_base_test \
  --output runs/custom_faults/my_test.json
```

For later evaluations, `--base-fixed-mechanisms` overrides the fixed-manifest path in the recipe. Use the same new manifest for every model. `--output` selects the evaluation output directory:

```bash
# Evaluate SFT and RL separately with the new fault assignment.
bash run.sh inference --config recipes/inference/sft_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_sft_test
bash run.sh inference --config recipes/inference/rl_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_rl_test
# Run paired clean/fault Base evaluation with that same assignment.
bash run.sh inference --config recipes/inference/base_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_base_comparison
# Optional: use the same assignment for the teacher reference.
bash run.sh inference --config recipes/inference/teacher_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_teacher_test
```

Omitting these arguments keeps the recipe's default paths. Overrides apply only to the current run, without rewriting recipes or supplied manifests. Use fresh output directories for a new assignment, and rerun every model under that same manifest for comparison.

For Val, use the corresponding assignment, freeze and evaluation recipes:

```bash
bash run.sh inference --config recipes/inference/base_fault_assignment_val.json \
  --output runs/my_base_val
python -m robustops.evaluation.reproduction \
  --action freeze --config recipes/data/freeze_val.json \
  --base-run runs/my_base_val \
  --output runs/custom_faults/my_val.json
bash run.sh inference --config recipes/inference/sft_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_sft_val
bash run.sh inference --config recipes/inference/rl_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_rl_val
# Select the same new manifest for the Base Val comparison.
bash run.sh inference --config recipes/inference/base_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_base_val_comparison
# Optional: use the same new manifest for teacher Val.
bash run.sh inference --config recipes/inference/teacher_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_teacher_val
```

## Use new labels for evaluation

To reassign faults along your own Base paths, reuse the supplied selector record and follow the [assignment and freezing flow above](#generate-and-use-a-new-fault-assignment). If you also rescan the selector, first freeze its results into the evaluation `policy.json`. This freezes tool labels and task inputs; the later Base run still selects the final mechanism.

For Test, copy the two configurations below. In the selector copy, set `output=runs/my_test_selector` and add `schedule=.local/test_schedule.json`, `tokenizer=models/base/Arctic-MidTool-RL-4B` and `policy_output=runs/my_test_selector/policy.json`. Retain the shared strategy in the schedule binding; the new Base run saves the caps of its selected mechanisms.

```bash
# Copy selector and schedule bindings, then edit as described above.
mkdir -p .local
cp recipes/data/fault_selector/test.json .local/test_selector.json
cp recipes/enterpriseops/faults/test.json .local/test_schedule.json
# Scan this split; valid existing results are reused.
python -m robustops.data.fault_candidates --config .local/test_selector.json --phase select
# Freeze selector results for evaluation without drawing caps or calling the teacher API again.
python -m robustops.data.fault_candidates --config .local/test_selector.json --phase freeze
```

The `freeze` phase reuses results saved by `select` above without another label request. For Val, copy the corresponding `val.json` files. Bind `selector_policy` in both the Base assignment and mechanism-freezing recipes to the new `policy.json`, with matching `schedule` and `base_run`; freezing reads the candidate limits saved by that Base run. Later Base/SFT/RL/teacher evaluation recipes use the new `selector_policy`; pass `--base-fixed-mechanisms` for the new fixed manifest and `--output` for a fresh output directory.

Default summaries read `runs/base_test`, `runs/sft_test` and `runs/rl_test` (and optionally `runs/teacher_fixed_test`). If you change evaluation output paths, update `runs` in the corresponding summary recipe.

## SFT Val Pass@4

See the [collection guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-collect-sft-val-pass4-trajectories) for four independent SFT Val samples, commands, resume behavior and outputs.

## Local configuration

Copy a recipe into `.local/` to change model paths, data or output directories. Filenames describe the task; learning rates, batches and steps stay inside JSON. SFT/RL validation loss is diagnostic and does not contribute parameter updates; see [Val data processing](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-small-val-monitoring-set) to prepare your own data. Default recipes read teacher products from `data/fault_selector/`, `data/verifier_graph/` and `data/sft/`; the [data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-generate-or-collect-your-own) explains how to generate and bind your own outputs.

Supplied Base/SFT/RL Val/Test recipes default to one 32 GB GPU and concurrency `2`. The recommended minimum for BF16 evaluation is one 16 GB GPU with `concurrency_per_gpu` set to `1`. Set GPU count with `hardware.gpus` or override it with `--gpus`; use `CUDA_VISIBLE_DEVICES` to select devices. Training still uses all visible GPUs when `--gpus` is omitted; `hardware.recorded` only describes the original experiment. For training and model-evaluation recipes, add `--dry-run` to inspect the plan without executing the run. Fault-selection dry runs prepare request files without calling the API. Use `--gpus N --dry-run` on a CPU machine.

`support/` and `enterpriseops/` contain runtime configuration. SFT and RL training directly export loadable Hugging Face model directories; see the [training guide](training/README.md) for export and usage details. Download instructions for published models are in [model assets](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-2-prepare-model-assets). Default evaluation uses BF16 RL from `models/rl`; original FP32 weights use `models/rl_fp32`.
