# Training guide

[English](README.md) · [简体中文](README.zh-CN.md)

Run these commands from the repository root. See the [main README](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-5-train) for methods and results.

## Start training

```bash
# Train SFT using all visible GPUs.
bash run.sh train --config recipes/training/sft.json
# Train RL from SFT through step 120.
bash run.sh train --config recipes/training/rl.json
```

The recommended minimum for training is **one GPU with 80 GB of VRAM**. Training recipes: [SFT](sft.json) · [RL](rl.json).

## GPUs and resuming training

For RL, GPU count changes the number of FSDP ranks and rollout workers, while each update retains 32 groups × 4 members. The group batch must be divisible by the GPU count (for example, 1/2/4/8). Full optimizer-checkpoint resume requires the same GPU count; to change it, initialize from an HF model export with a new optimizer.

RL resource settings live in the `training` object. `ray_num_cpus` and `ray_object_store_memory_gib` default to `null`, allowing Ray to detect available resources; set them explicitly to assign a budget. `min_host_memory_gib` is an optional minimum, disabled at `0`. `rollout_gpu_memory_utilization`, `rollout_max_num_batched_tokens` and `actor_token_batch_per_gpu` control inference and training allocations.

`max_context_tokens`, `max_generation_tokens` and `max_turns` set the RL context limit, per-turn generation limit and turn limit. The defaults remain 32,768, 2,048 and 50. The context limit also controls model serving, fault suffix eligibility and SFT validation during RL. For SFT training, use `training.max_length` and `training.max_token_len_per_gpu`.

## Reward normalization configuration

Set these fields in the `training` object of `rl.json`:

```json
{
  "advantage_estimator": "grpo",
  "reward_normalization": "component",
  "reward_weights": [1.0, 0.3, 0.1]
}
```

- `aggregate`: weight and sum success, progress and quality rewards, then standardize within each four-trajectory group.
- `component`: standardize each component within its group, then apply the same weights. This is the default.

Both methods apply equal-trajectory batch whitening followed by the same local-credit and loss-weight paths. Weights are ordered as success, progress, quality. Each configuration launches one run under `runs/rl/my_run/`, with rollout logs and checkpoints in its `trainer/` directory. Compare the normalization methods using separate configuration files.

## Resume and fork

Replace `runs/rl/my_run` below with an actual run directory. `--to-step` is the cumulative target step:

```bash
# Resume the latest full checkpoint with its saved training configuration.
python -m robustops.training.rl_launcher \
  --resume-run runs/rl/my_run --to-step 150 --gpus 2

# Change reward_normalization in a recipe copy and fork retained step 30.
python -m robustops.training.rl_launcher \
  --config .local/rl_aggregate.json --resume-run runs/rl/my_run \
  --resume-step 30 --fork-output runs/rl/aggregate_fork --to-step 120 --gpus 2
```

Both commands support `--dry-run`. Ordinary resume permits changes to duration, saving, resource limits and timeouts; changing rewards or local credit requires a fork. A fork loads the source model, optimizer, RNG and sampler state, copies task data and initial states, and writes new logs and checkpoints into a new directory. It does not modify the source run. Seed, group size, task-sampling mode and optimizer learning rate must match the source checkpoint.

Alternatively, add `"resume": {"run": "runs/rl/my_run"}` at the JSON top level, change `training.steps`, and use `run.sh train --config ...`. For a fork, add `step` and `fork_output` to `resume`; omitting `step` selects the latest checkpoint. Resume requires full training state saved by this interface and the same GPU count. HF weights alone do not contain the optimizer or sampler cursor.

## Checkpoint saving

RL saves every 10 steps by default through `training.checkpoint_save_frequency`. SFT uses `checkpoint.save_frequency`, either a positive step interval or `"after_each_epoch"`; the final training step is also saved.

Set SFT's `checkpoint.max_to_keep` or RL's `training.checkpoint_max_to_keep` to a positive integer N to retain only the latest N checkpoints in that run's checkpoint directory. Omit the setting or use `null` to retain all checkpoints. Older checkpoints are removed after a new complete save, including when continuing in a directory containing earlier checkpoints. There are no permanently protected steps. The supplied recipes explicitly retain 3 SFT checkpoints and 1 RL checkpoint.

The supplied SFT recipe uses `save_contents: ["hf_model"]` to export weights, configuration and tokenizer for inference; it does not preserve optimizer state. For full training continuation, save `["model", "optimizer", "extra"]`; add `"hf_model"` if a standalone HF export is also required. Dataloader state is saved alongside the selected contents. RL's separate HF exports in `trainer/models/` are not deleted by full-checkpoint retention.

## Local model export

Training directly saves complete Hugging Face model directories containing weights, model configuration and tokenizer files for local inference or further training.

- SFT: each epoch saves to `checkpoints/global_step_<step>/huggingface/` within the run directory.
- RL: `training.hf_export_steps` selects the steps at which FSDP weights are merged and exported; the defaults are 30, 100 and 120. Exports are saved to `trainer/models/global_step_<step>/` within the run directory.

Current exports preserve FP32 weights and can be loaded with BF16 compute for evaluation without a separate precision conversion. Optimizer and sampling state for full resume remain in the training checkpoints.

## Evaluate retrained weights

After training, bind the step-120 HF export as `models/rl_fp32` and copy the Test recipe:

```bash
# Prepare a local evaluation recipe for the retrained weights.
mkdir -p .local
cp recipes/inference/rl_test.json .local/rl_test_from_training.json
```

In the copy's `models` list, set the `step120` entry to `path: "models/rl_fp32"` and `weight_dtype: "float32"`. Run evaluation with this configuration:

```bash
bash run.sh inference --config .local/rl_test_from_training.json
```

Both FP32 and BF16 weights run with BF16 compute.

## Advantage and policy-loss implementation

The released recipe uses three raw reward components: success, progress and quality. Group normalization and batch whitening produce relative advantages; local credit then assigns them to generated tokens. These steps are implemented separately from final loss aggregation.

| Stage | Implementation and current behavior |
|---|---|
| Rewards | `components` in [rl_credit.py](../../src/robustops/training/rl_credit.py) computes S, P and Q. P measures net progress from the continuation boundary; Q measures call quality in successful trajectories. |
| Advantages | `advantages` in [rl_advantage.py](../../src/robustops/training/rl_advantage.py) selects aggregate-first or component-first normalization within G=4, then whitens scalar trajectory advantages across the accepted batch with equal statistical weight per trajectory. |
| Token credit | `credit_segments` in [recovery_local_credit.py](../../src/robustops/training/recovery_local_credit.py) allocates Q credit over time before overriding advantages in bad-action and failed-stop spans. Token advantages can differ within one trajectory. |
| Flow weights | `flow_multipliers` in [rl_action_feedback.py](../../src/robustops/training/rl_action_feedback.py) sets `(n_s/N) * T/T_s` for flow s; [recovery_rl_alignment.py](../../src/robustops/training/recovery_rl_alignment.py) stores it in `recovery_loss_weights`. |
| Loss | The [veRL patch](../../scripts/verl.patch) connects these advantages and weights. `workers/utils/losses.py` applies the weights and `core_algos.agg_loss` averages over global valid tokens. With 64 clean and 64 fault trajectories, each flow's token mean receives weight 1/2. |

The [launcher](../../src/robustops/training/rl_launcher.py) sets `algorithm.adv_estimator=grpo`, with the project patch supplying the reward normalization and local-credit implementation described above. `loss_agg_mode=token-mean` selects aggregation, while `policy_loss.loss_mode=vanilla` selects the PPO clipping objective. Current symmetric clipping is 0.2/0.2, with negative-advantage dual-clip at 3. Reference-policy KL has coefficient 0.001, uses `low_var_kl` and is added directly to the loss with the same flow weights. KL is outside the raw rewards; the entropy coefficient is zero.

FSDP determines the global denominator before splitting microbatches. It counts only newly generated assistant tokens with `response_mask=1`; tool observations, inherited prefixes and padding contribute no loss. Equal trajectory weighting in advantage whitening does not imply equal trajectory weighting in the loss. Group comparison, batch whitening and token-level aggregation are distinct operations.

$$
L_{\mathrm{PG}}=
\frac{1}{2T_{\mathrm{clean}}}\sum_{(i,t)\in\mathrm{clean}}\ell_{i,t}
+\frac{1}{2T_{\mathrm{fault}}}\sum_{(i,t)\in\mathrm{fault}}\ell_{i,t}.
$$

`ℓ` is the PPO-clipped token policy loss; `T_clean` and `T_fault` count valid generated assistant tokens in each flow. KL uses the same flow weights and token averaging.

## Target behaviors for local credit

Local credit applies to action tokens newly generated by the current policy. Accepted training groups first receive S/P/Q advantages and temporally allocated quality credit; the rules below then override local advantages. History preceding the recovery boundary may supply evidence, but its tokens are not trained or penalized again.

| Target behavior | Triggering evidence | Current treatment |
|---|---|---|
| Repeating explicitly rejected arguments (`repeated_invalid_arguments`) | The same tool and arguments receive the same visible argument/schema rejection again | Apply `min(A, -0.5)` to that tool-call span |
| Reusing a known unexposed tool (`repeated_unexposed_tool`) | Explicit feedback established that the tool is unavailable, and the agent calls it again | Apply `min(A, -0.5)` to the repeated call |
| Reusing an explicitly missing resource reference (`repeated_reference_rejection`) | Feedback identifies the argument field and missing reference; the same call receives the same rejection again, without intervening observations or changes that invalidate the evidence | Apply `min(A, -0.5)` to the repeated call |
| Reading repeatedly without progress (`repeated_no_progress_read`) | A confirmed read-only tool returns the same complete result for the same arguments, without a new query or observation; marking starts at the third occurrence | Apply `min(A, -0.5)` to the marked read’s tool-call span |
| Stopping before task success (`failed_native_stop`) | The trajectory ends with a native model stop, its final turn makes no tool call, and the success checks do not all pass | Apply `min(A, 0)` only to the final assistant stop span, removing positive advantage while retaining existing negative advantage |

Tool-call overrides cover exactly `<tool_call>` through `</tool_call>`, leaving accompanying narration and earlier useful actions outside the override. Failed-stop overrides cover the final newly generated assistant span. Here `A` is the advantage at that position before the override, rather than a trajectory reward from which a fixed cost is subtracted.

The first three rules require earlier explicit error evidence; a first failure alone does not trigger these repetition penalties. A generic 404/5xx does not establish a bad resource reference. Injected errors from legitimate calls are excluded from native bad-action evidence, preserving legitimate retries and recovery as learning opportunities.

The repeated-read rule allows two identical observations. A new query or changed result renews the verification allowance; writes, unknown tools, errors, injected responses and incomplete results clear its accumulated evidence. Unfinished pagination or a still-running asynchronous operation does not count as an unchanged complete result. Alternating between the same A and B results does not automatically clear repetition records.

The failed-stop rule removes positive encouragement for ending an unfinished task. It does not inspect whether the text claims success or establish that continuation would help. Budget or context exhaustion is outside this native-stop rule; normal stops after task success are unaffected.

The first three repeat-error categories also contribute to the bad-call count in quality reward `Q`. Repeated reads and failed stops affect local credit only and do not enter that count. Failed trajectories have raw `Q = 0`, but their group-normalized quality advantage can still be nonzero.

Implementation: [repeat-error evidence](../../src/robustops/training/rl_action_feedback.py) · [fault exclusions and span assembly](../../src/robustops/training/rl_agent.py) · [repeated reads, stopping and advantage overrides](../../src/robustops/training/recovery_local_credit.py) · [exact call spans](../../src/robustops/training/rl_credit.py). See `local_bad_eta`, `local_read_repeat_from` and `terminal_stop_advantage_cap` in the [RL recipe](rl.json). Teacher-data masking is described in [SFT loss masks](https://www.github.com/haohuiliang/RobustOps/blob/main/data/sft/README.md#user-content-teacher-export-loss-masks-and-target-behaviors).

## Custom model and data inputs

RL accepts a local HF safetensors export through `model.path`; it is not restricted to the published SFT step or epoch. Required model/tokenizer files, weight payloads and shard indexes are checked. FP32, FP16 and BF16 exports retain their stored precision. `model.lineage` is optional descriptive provenance; actual export metadata and its hashes are recorded in `model_provenance.json`.

The top-level `data` object in `rl.json` selects the task manifest, training pool, fault selector, SFT records/provenance/exposure, SFT validation Parquet and validation recipe/policy. Paths are relative to the project root or absolute. Omitted keys retain the recipe defaults. Set `sft_provenance` or `sft_fault_exposure` to `null` when using the other exposure source. `validation_policy: null` uses the policy referenced by `validation_recipe`.

Counts come from the supplied manifests. The pool must cover all runnable Train task/mode cells in the supplied manifest, with at least 32 pairs for the 32-group batch size. IDs, domains, modes, selector coverage and Train/Val separation are checked. Historical selector entries outside this manifest are ignored. New runs record input file hashes; resume, fork and task-plan reuse reject changed inputs. SFT validation accepts any nonempty dataset and excludes worker padding from its record average.

For evaluation, a model entry can use `{"key": "sft", "sft_train_run": "runs/sft/my_run", "step": 120}`. Omit `step` to read `latest_checkpointed_iteration.txt` in the run's configured checkpoint directory. A missing marker requires an explicit step.
