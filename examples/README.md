# Five Base / RobustOps examples

[English](README.md) · [简体中文](README.zh-CN.md)

Run two clean tasks and three fault tasks without running the full evaluation. Each example uses a fixed task, seeded database, tool set and state verifier. Both models start from independently reset task state. These are selected demonstrations, not a representative benchmark sample; see the [full evaluation](../README.md#6-evaluation) for aggregate results.

## Setup

Follow the repository's [environment and runtime setup](../README.md#quick-start). Download the [Base model and RL BF16 model](../README.md#models) into `models/base/Arctic-MidTool-RL-4B` and `models/rl`. No SFT model or teacher API key is needed. Run commands from the repository root with the `rops` environment activated.

The default uses one GPU, one active trajectory at a time, and runs Base then RL sequentially. Use `CUDA_VISIBLE_DEVICES` to select a GPU. A 32 GB GPU is recommended for the 32K context budget. The runner starts the required MCP services and stops its model and MCP processes afterwards.

```bash
# Inspect all five task bindings without starting model/MCP services.
bash scripts/demo.sh --dry-run
# Run both models on all five examples, then generate the comparison report.
bash scripts/demo.sh
# Run one example with both models into a separate directory.
bash scripts/demo.sh --case calendar_retry --output runs/demo_calendar
# Run one model at a time into the same comparison directory.
bash scripts/demo.sh --model base --output runs/demo_separate
bash scripts/demo.sh --model rl --output runs/demo_separate
# Rebuild reports from saved results without loading models or starting MCP.
bash scripts/demo.sh --report-only
```

Repeat `--case` to select several examples. Use a fresh `--output` when changing the selection or configuration. Re-running the same plan reuses valid saved attempts; it does not retry failures to obtain a successful demonstration. Preserve invalid attempts and choose a new output directory after fixing their cause.

## Examples

| ID | Condition | Task | Fault |
|---|---|---|---|
| `teams_channels` | Clean | Create an engineering channel if missing, then archive an old partner channel | None |
| `drive_access` | Clean | Approve a pending document access request and update collaboration settings | None |
| `itsm_reply` | Fault | Correct a device's cost and create a high-impact incident | One eligible read receives a mismatched reply |
| `calendar_retry` | Fault | Create a calendar, set its properties and schedule a meeting | Up to four eligible writes are rejected before execution |
| `hr_filter` | Fault | Reassign ready, critical-priority HR cases and align ownership | One eligible read ignores an optional filter |

Faults apply at the earliest eligible calls; models can choose different tools, so identical fault rules do not imply identical call positions. A configured fault with no actual hit is explicitly marked `NO FAULT HIT`.

These prompts refer to objects in the supplied databases. Editing a prompt alone does not define a new evaluated task: the task data, tools and verifier must agree with it. To inspect the full task request, use the generated report or saved `plan.json`.

## Outputs

Open `runs/demo/comparison.txt` in a text editor, or read it in the terminal:

```bash
less runs/demo/comparison.txt
```

The script finishes by printing this file's path. The file starts with a short Base / RL results table, followed by each task's failed checks, fault positions, final-response excerpts and termination reasons. For the complete task, final responses and tool traces, open the corresponding `details/<case_id>.txt`. Long tool arguments and responses in those details are abbreviated; the raw JSON records retain them in full.

Complete records remain in `<output>/{base,rl}/<case_id>/{clean,fault}/attempt.json` and `calls/`. `comparison.json` contains structured results; each model directory retains its resolved plan and runtime logs.

`PASS` requires valid execution, successful database cleanup, all task checks, and an actual injection for fault examples. `FAIL` is a valid task failure; `INVALID`, `NO FAULT HIT` and `NOT RUN` remain distinct. Passing state checks does not guarantee a final response or efficient stopping, so termination and call counts are displayed separately.

The example definitions are in [demo.json](../data/cases/demo.json). [Base](../recipes/inference/demo_base.json) and [RL](../recipes/inference/demo_rl.json) recipes define the model paths and execution settings. Both use the ordinary case runner.
