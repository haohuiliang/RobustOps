# 训练指南

[English](README.md) · [简体中文](README.zh-CN.md)

以下命令从仓库根目录运行，方法与结果见主文档（[README](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-5-训练)）。

## 启动训练

```bash
# 使用全部可见 GPU 训练 SFT。
bash run.sh train --config recipes/training/sft.json
# 从 SFT 训练 RL 至 step 120。
bash run.sh train --config recipes/training/rl.json
```

训练建议最低配置为**单卡 80 GB 显存**。训练配置：[SFT](sft.json) · [RL](rl.json)。

## GPU 与续训

RL 的 GPU 数量决定 FSDP 进程（`rank`）与采样工作进程（`rollout worker`）数，每步仍保持 32 组 × 4 条。组批量须能被卡数整除，例如 1/2/4/8 卡。完整优化器检查点续训要求卡数相同；换卡数时从 HF 模型权重开始，重新初始化优化器。

RL 资源参数位于 `training` 对象。`ray_num_cpus` 和 `ray_object_store_memory_gib` 默认为 `null`，由 Ray 检测可用资源，也可填写明确的资源配额。`min_host_memory_gib` 是可选的主机内存下限，`0` 表示不设下限。`rollout_gpu_memory_utilization`、`rollout_max_num_batched_tokens` 和 `actor_token_batch_per_gpu` 控制推理及训练的资源分配。

`max_context_tokens`、`max_generation_tokens` 和 `max_turns` 分别设置 RL 上下文长度、每轮生成长度和最大回合数，默认仍为 32,768、2,048 和 50。上下文长度统一用于模型服务、故障恢复后缀位置检查及 RL 中的 SFT 验证。SFT 训练则使用 `training.max_length` 和 `training.max_token_len_per_gpu`。

## 奖励归一化配置

`rl.json` 的 `training` 中使用以下字段：

```json
{
  "advantage_estimator": "grpo",
  "reward_normalization": "component",
  "reward_weights": [1.0, 0.3, 0.1]
}
```

- `aggregate`：成功、进展、质量按权重相加后，在同组四条轨迹内标准化。
- `component`：各奖励分别在组内标准化，再按相同权重相加，默认使用此方式。

两种方式都继续对接受批次的轨迹标量做等权白化，然后应用相同的局部信用和 loss 权重。权重依次对应成功、进展、质量。每份配置启动一个独立运行，输出在 `runs/rl/my_run/`；采样日志和检查点位于其 `trainer/` 子目录。在独立的配置副本中修改归一化方式，即可进行对照。

## 恢复与分叉

下面的 `runs/rl/my_run` 替换为实际运行目录，`--to-step` 是累计目标步数：

```bash
# 从最近一次完整检查点恢复，沿用保存的训练配置。
python -m robustops.training.rl_launcher \
  --resume-run runs/rl/my_run --to-step 150 --gpus 2

# 在配置副本中修改 reward_normalization，再从保留的 step 30 分叉。
python -m robustops.training.rl_launcher \
  --config .local/rl_aggregate.json --resume-run runs/rl/my_run \
  --resume-step 30 --fork-output runs/rl/aggregate_fork --to-step 120 --gpus 2
```

两条命令都支持 `--dry-run`。普通续训只允许延长步数或调整保存、资源和超时设置；改变奖励或局部信用须分叉。分叉加载源检查点的模型、优化器、随机状态和采样队列，复制任务数据与初始状态，将新日志和检查点写入新目录。源运行不修改。种子、组大小、任务采样方式和优化器学习率须与源检查点一致。

也可在 JSON 顶层加入 `"resume": {"run": "runs/rl/my_run"}`，修改 `training.steps`，继续使用 `run.sh train --config ...`。分叉在 `resume` 中增加 `step` 和 `fork_output`；省略 `step` 时使用最新检查点。恢复要求此接口保存的完整训练状态及相同 GPU 数量；只有 HF 权重不包含优化器和采样游标，不能作为完整续训检查点。

## 检查点保存

RL 默认通过 `training.checkpoint_save_frequency` 每 10 步保存一次。SFT 使用 `checkpoint.save_frequency`，支持正整数步数间隔或 `"after_each_epoch"`；训练最后一步也会保存。

SFT 的 `checkpoint.max_to_keep` 和 RL 的 `training.checkpoint_max_to_keep` 可设为正整数 N，只保留该运行 checkpoint 目录中最近 N 个检查点。省略或设为 `null` 时全部保留。新检查点完整保存后才淘汰最旧的版本；在已有检查点的目录中恢复训练时，旧版本也计入上限。不再有指定步数永久保留的例外。仓库提供的配置显式设置 SFT 保留 3 个、RL 保留 1 个。

仓库提供的 SFT 配置使用 `save_contents: ["hf_model"]` 导出权重、模型配置和 tokenizer，供推理使用，不包含优化器状态。完整续训需保存 `["model", "optimizer", "extra"]`；如同时需要独立 HF 导出，再加入 `"hf_model"`。数据加载器状态会随所选内容一起保存。RL 在 `trainer/models/` 下单独生成的 HF 导出不受完整 checkpoint 保留数量限制。

## 本地模型导出

训练直接保存完整的 Hugging Face 模型目录，包含权重、模型配置和 tokenizer，可供本地推理或后续训练使用。

- SFT：每个 epoch 结束时保存到该运行的 `checkpoints/global_step_<步数>/huggingface/`。
- RL：在 `training.hf_export_steps` 指定的步骤合并 FSDP 权重并导出，默认步骤为 30、100、120。导出保存到该运行的 `trainer/models/global_step_<步数>/`。

当前导出保留 FP32 权重，评估时可直接以 BF16 加载，无需另做精度转换。完整续训所需的优化器与采样状态保存在训练检查点中。

## 评估重训权重

训练后将第 120 步 HF 导出绑定为 `models/rl_fp32`，再复制 Test 配置：

```bash
# 为重新训练的权重准备本地评估配置。
mkdir -p .local
cp recipes/inference/rl_test.json .local/rl_test_from_training.json
```

在副本的 `models` 列表中，将 `step120` 项设为 `path: "models/rl_fp32"`、`weight_dtype: "float32"`，使用该配置运行评估：

```bash
bash run.sh inference --config .local/rl_test_from_training.json
```

FP32 与 BF16 权重均以 BF16 计算。

## 优势与策略损失的实现路径

提供的训练配置使用成功、进展、质量三个原始奖励分项。组内归一化与批次白化生成相对优势，局部信用再将其分配到生成 token；这些步骤与最后的 loss 聚合分别实现。

| 环节 | 实现与当前行为 |
|---|---|
| 奖励 | [rl_credit.py](../../src/robustops/training/rl_credit.py) 的 `components` 计算 S、P、Q；P 衡量相对续行起点的净进展，Q 衡量成功轨迹的调用质量。 |
| 优势 | [rl_advantage.py](../../src/robustops/training/rl_advantage.py) 的 `advantages` 按 `reward_normalization` 选择合并或分项归一化，再对接受批次的轨迹标量白化。每条轨迹在白化统计中权重相同。 |
| token 信用 | [recovery_local_credit.py](../../src/robustops/training/recovery_local_credit.py) 的 `credit_segments` 先分配 Q 的时序信用，再覆盖坏片段和失败停止片段的优势；同一轨迹的 token 优势不必相同。 |
| 分流权重 | [rl_action_feedback.py](../../src/robustops/training/rl_action_feedback.py) 的 `flow_multipliers` 为流 s 设置 `(n_s/N) * T/T_s`，由 [recovery_rl_alignment.py](../../src/robustops/training/recovery_rl_alignment.py) 写入 `recovery_loss_weights`。 |
| loss | veRL 补丁（[verl.patch](../../scripts/verl.patch)）接入上述优势和权重；`workers/utils/losses.py` 施加权重，`core_algos.agg_loss` 按全局有效 token 数平均。当前 clean/fault 各 64 条，等价两类各自 token-mean 再各乘 1/2。 |

启动配置（[rl_launcher.py](../../src/robustops/training/rl_launcher.py)）设置 `algorithm.adv_estimator=grpo`，通过项目补丁接入上述奖励归一化与局部信用实现。`loss_agg_mode=token-mean` 与 `policy_loss.loss_mode=vanilla` 分别控制聚合方式和 PPO 裁剪目标。当前对称 clip 为 0.2/0.2，负优势 dual-clip 为 3；参考策略 KL 系数 0.001，以 `low_var_kl` 直接加入 loss，并采用相同分流权重。KL 不进入原始奖励，熵项系数为 0。

全局分母在 FSDP microbatch 切分前确定，只包含 `response_mask=1` 的新生成助手 token。工具观测、继承前缀和 padding 不贡献 loss。优势白化中的轨迹等权不意味着 loss 轨迹等权；组内相对比较、批次白化和 token 级聚合是三个不同操作。

$$
L_{\mathrm{PG}}=
\frac{1}{2T_{\mathrm{clean}}}\sum_{(i,t)\in\mathrm{clean}}\ell_{i,t}
+\frac{1}{2T_{\mathrm{fault}}}\sum_{(i,t)\in\mathrm{fault}}\ell_{i,t}.
$$

其中 `ℓ` 为经 PPO 裁剪的 token 策略损失，`T_clean`、`T_fault` 分别为两类有效生成助手 token 的总数。KL 使用相同的两类权重与 token 平均。

## 局部信用的目标行为

局部信用作用于当前策略新生成的动作 token。已接受训练组先计算 S/P/Q 优势与质量信用的时间分配，再按下表覆盖局部优势。恢复起点之前的历史可以提供判定证据，但不重复训练或处罚这些历史 token。

| 目标行为 | 触发证据 | 当前处理 |
|---|---|---|
| 重复提交已被明确拒绝的参数（`repeated_invalid_arguments`） | 相同工具与参数再次得到相同的、可见的参数/schema 错误反馈 | 对该工具调用片段施加 `min(A, -0.5)` |
| 继续使用已知未暴露的工具（`repeated_unexposed_tool`） | 已明确收到工具未暴露反馈，随后再次调用该工具 | 对再次调用的片段施加 `min(A, -0.5)` |
| 重复使用已明确不存在的资源引用（`repeated_reference_rejection`） | 错误明确指出参数字段与缺失引用；同一调用再次得到相同拒绝，期间没有使原证据失效的新观察或状态变化 | 对再次调用的片段施加 `min(A, -0.5)` |
| 无进展的重复读取（`repeated_no_progress_read`） | 已确认只读的工具以相同参数返回相同完整结果，期间没有新查询或新观察；从第三次起标记 | 对已标记读取的工具调用片段施加 `min(A, -0.5)` |
| 任务未成功便主动结束（`failed_native_stop`） | 当前轨迹以模型主动结束收尾，最后一轮没有工具调用，且成功检查未全部通过 | 仅对最后助手结束片段施加 `min(A, 0)`：撤掉正优势，已有负优势保持 |

工具调用的覆盖范围精确到 `<tool_call>` 至 `</tool_call>`，不连带覆盖同轮解释或此前有效动作；失败结束覆盖最后一轮新生成的助手片段。公式中的 `A` 是覆盖前该位置的优势，不是从轨迹奖励中再减去固定分数。

前三类需要已有明确错误证据；第一次失败本身不触发这三类重复处罚。一般的 404/5xx 不足以证明资源引用错误。合法调用遇到注入故障时，注入错误不作为原生坏动作证据；合理重试和恢复仍可学习。

重复读取允许两次相同观察。新查询或变化的结果会重新给予核验机会；写入、未知工具、错误、注入返回和不完整结果会清除该规则的累计证据。分页未完、异步任务仍在运行等情况不按“相同完整结果”处罚。交替读取同样的 A、B 结果不会自动清空重复记录。

失败结束规则只撤销对未完成任务的结束动作的正鼓励，不判断自然语言是否声称成功，也不保证继续执行一定有收益。预算或上下文耗尽不是这里的模型主动结束；任务成功后的正常结束不受此规则影响。

前三类重复错误还计入调用质量 `Q` 的坏调用数；无进展读取和失败结束只影响各自的局部信用，不加入该计数。全失败轨迹的原始 `Q` 为零，但组内归一化后的质量优势仍可能非零。

实现：重复错误证据（[rl_action_feedback.py](../../src/robustops/training/rl_action_feedback.py)） · 故障排除与片段组装（[rl_agent.py](../../src/robustops/training/rl_agent.py)） · 重复读取、停止与优势覆盖（[recovery_local_credit.py](../../src/robustops/training/recovery_local_credit.py)） · 精确调用区间（[rl_credit.py](../../src/robustops/training/rl_credit.py)）。参数见 RL 配置（[rl.json](rl.json)）的 `local_bad_eta`、`local_read_repeat_from`、`terminal_stop_advantage_cap`。教师数据的零损失处理见 [SFT loss mask](https://www.github.com/haohuiliang/RobustOps/blob/main/data/sft/README.zh-CN.md#user-content-教师导出的-loss-mask-与目标行为)。

## 自定义模型与数据

RL 通过 `model.path` 接受本地 HF safetensors 导出，支持不同步数或轮次的 SFT 检查点。启动前检查模型与 tokenizer 文件、权重完整性及分片索引，允许保持原存储精度的 FP32、FP16、BF16 导出。`model.lineage` 可选，仅作为来源说明；实际模型元数据及其哈希记录到 `model_provenance.json`。

`rl.json` 顶层 `data` 对象可指定任务清单、训练池、故障选择器、SFT 记录及来源/故障暴露清单、SFT 验证 Parquet、验证配置与策略。路径相对项目根目录或使用绝对路径，未填写的字段沿用默认路径。使用另一种故障暴露来源时，可将 `sft_provenance` 或 `sft_fault_exposure` 设为 `null`；`validation_policy: null` 表示使用 `validation_recipe` 引用的策略。

数据数量从输入清单读取。训练池须覆盖所提供任务清单中全部可运行 Train 任务×模式组合，每步 32 组的批次设置要求至少 32 个组合。保留 ID、业务域、模式、选择器覆盖及 Train/Val 隔离检查；选择器中不属于该清单的历史条目不参与训练。新运行记录输入文件哈希，续训、分叉及任务计划复用会拒绝已变更的输入。SFT 验证集支持任意非空规模，计算记录平均损失时排除 worker 补齐项。

评估模型可配置为 `{"key": "sft", "sft_train_run": "runs/sft/my_run", "step": 120}`。省略 `step` 时，读取该运行所配置检查点目录中的 `latest_checkpointed_iteration.txt`；缺少此标记时须显式指定步数。
