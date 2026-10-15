# 数据资产

[English](README.md) · [简体中文](README.zh-CN.md)

仓库直接提供以下数据；默认配置可直接使用三类模型辅助产物，也支持自行生成或采集。

| 目录 | 内容 |
|---|---|
| [fault_selector/](fault_selector/) | 选择器扫描 Train/Val/Test 后的逐工具故障标签：`train.jsonl`、`val.jsonl`、`test.jsonl` |
| [verifier_graph/](verifier_graph/) | 教师生成的 Train/Val 检验依赖图，用于 RL 进度反馈 |
| [sft/](sft/) | 692 条 Train 教师成功示范，以及用于快速监控 Val 集表现的 15 条小数据，JSONL/Parquet |
| [task_sets/](task_sets/) | 固定 Train/Val/Test 任务、默认 80% 采集清单与历史 SFT/RL 池清单 |

`fault_selector/` 保存故障选择器扫描数据集的结果，列出每个任务×工具配置下各工具可能适用的故障。LLM 生成的标签用于缩小运行时注入器的检查范围，只检查候选工具×故障组合；注入器再根据无注入（`clean`）轨迹中的真实调用、返回和状态核验是否可注入。Test 的 130 条工具标签与固定 Test 选择器记录（[policy.json](../artifacts/test_fault_policy/policy.json)）一致；评估再读取 Base 固定机制与预算（[base_test_fixed_mechanisms.json](../artifacts/base_test_fixed_mechanisms.json)），两者职责不同。完整注入流程（[Fault injection](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-故障注入流程)）

Train 文件保留扫描批次的 1,027 条标签；当前采集与训练按有效任务清单取用其中 1,025 个可运行组合。Val/Test 分别为 127/130 条。选择器结果包含 Test，不代表 Test 会进入训练。

依赖图用于 RL 的部分进度反馈，完整成功仍由任务状态检查判定。SFT 示范保留真实执行上下文，屏蔽已识别无效决策的目标 loss；合法故障遭遇和有效恢复仍可学习。筛选条件失效（`ignored_filter`）整个机制族不参与本项目 SFT/RL，用于检验对未见故障机制的泛化能力；训练故障记录见 [fault_exposure.json](sft/train/fault_exposure.json)。

有效总体为 649 个公开 Oracle 任务 + 2 个可执行的上游 revised 补充任务 − 3 个因任务定义问题排除的任务 = **648 个任务**：Train 518、Val 64、Test 66。总体元数据（[enterpriseops.json](task_sets/enterpriseops.json)）记录排除任务的 ID 与原因，它们不属于任何有效分组。原始 `SOURCE_AUDIT.json` 保留早期来源审计记录，当前有效总体以 `task_sets/enterpriseops.{json,jsonl}` 为准。

## 自行生成或采集

先完成根 README 第 1–3 节的准备，并按教师 API 配置（[Teacher API](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-教师-api-配置)）选择填写 `.env.deepseek` 或设置 `DEEPSEEK_API_KEY` 环境变量；两者都有时环境变量优先。以下命令均在仓库根目录执行，生成结果写入 `runs/`，保留提供的数据。

### 自行生成故障标签

```bash
# 扫描 Train/Val/Test，生成各工具可能适用的故障标签。
bash run.sh inference --config recipes/inference/fault_selector.json
```

完整结果写入 `runs/fault_selector/dataset/{train,val,test}.jsonl`。选择器请求失败或结构不合格的项，可再次执行同一命令补齐；已有有效结果会复用，不按后续任务成绩反复改标签。

要独立启动新一批扫描，复制各分区与导出配置：

```bash
mkdir -p .local/fault_selector
cp recipes/data/fault_selector/{all,train,val,test}.json .local/fault_selector/
```

将各分区副本的 `output` 改为 `runs/my_fault_selector/train`、`runs/my_fault_selector/val`、`runs/my_fault_selector/test`；在 `all.json` 副本中将 `splits` 指向这些配置副本，并将 `output`、`staging`、`backup` 分别设为 `runs/my_fault_selector/dataset`、`runs/my_fault_selector/staging`、`runs/my_fault_selector/previous_dataset`。随后运行：

```bash
python -m robustops.data.fault_dataset --config .local/fault_selector/all.json
```

新标签位于 `runs/my_fault_selector/dataset/{train,val,test}.jsonl`。

教师采集使用新标签时，在本地 `recipes/data/teacher.json` 副本中把 `selector` 指向新的 `train.jsonl`。Val/Test 评估还需要将选择器记录整理为 `policy.json`，再由 Base 分配运行生成故障配置，不能直接把 JSONL 填进 `selector_policy`；转换和配置方法见将新标签用于评估（[Evaluation with new labels](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-将新标签用于评估)）。

### 自行生成检验依赖图

```bash
# 生成 Train/Val 检验依赖图。
python scripts/build_verifier_graphs.py --config recipes/data/verifier_graph.json \
  --run-dir runs/my_verifier_graph --output runs/my_verifier_graph_dataset
```

上述新批次写入 `runs/my_verifier_graph_dataset/`；省略覆盖参数时使用配置中的默认目录，在本地 RL 配置中修改 `training.verifier_graph_dir` 即可使用。检验依赖图生成是独立流程，不参与故障机制或注入位置的选择。

### 自行采集 SFT 示范

```bash
# 使用所选 Train 清单，采集、验收并导出 SFT 示范。
python -m robustops.data.teacher_collection \
  --config recipes/data/teacher.json --stage all
```

采集先收齐温度 0 的全部 clean，并保存真实合格调用前的状态和对话，再从已有机会中均衡选择故障机制，按训练与推理共用的策略配置（[strategy.json](../recipes/enterpriseops/faults/strategy.json)）抽取重复次数上限，直接从 clean 保存点开始注入和续行。采集仅使用可运行 Train 任务，排除 `ignored_filter`；没有合格机会的案例保留零注入。

业务失败的案例以温度 0.6 补采一次。

采集结束后自动调用统一的轨迹转 SFT 模块（[teacher_sft.py](../src/robustops/data/teacher_sft.py)）：去掉教师指导，保留完整执行历史，生成监督掩码 与 JSONL/Parquet。已识别的无效决策不作为学习目标；合法调用遇到暂态故障及有效恢复仍可学习。处理已有推理产物及加入审查标注的方法见下文。

另开一批采集时，将 `recipes/data/teacher.json` 复制到 `.local/teacher.json`，修改 `output` 与 `export` 为新目录，再把上面的 `--config` 改为该副本；选择其他 Train 范围的完整示例见下节。

原始记录写入 `runs/teacher_collection/`，通过任务成功和学生上下文验收后导出到 `runs/teacher_dataset/`，实际数量可能不同于 692。在本地 SFT 配置中修改 `data.train_parquet`、`data.expected_train_records` 与相应训练步数。修改的配置放在 `.local/`，采集范围见下节。

### 选择教师采集清单

默认使用 [sft_collection_pool.json](task_sets/sft_collection_pool.json)，即提供数据采用的 80% Train 范围。全量采集可修改本地配置的 `pool`，或用 `--pool` 指定其他同格式清单；例如 [recovery_train_pool.json](task_sets/recovery_train_pool.json) 会取其中标为 `sft` 的子集。教师案例使用本地案例配置中的 `teacher.pool`。

按自选比例生成新清单：

```bash
# 示例：按 Train 任务 ID 选择 60%，比例可自行修改。
python -m robustops.data.collection_pool \
  --fraction 0.6 --seed 20260916 --output .local/sft_pool.json
```

生成器只从当前任务清单的可运行 Train 任务中取样，兼顾业务域与工具配置；`--manifest` 可指定其他同格式任务 JSONL。比例取 `(0, 1]`，原有划分和提供的清单保持不变。新清单独立生成；复现本项目的采集范围时使用提供的 80% 清单。

将 `recipes/data/teacher.json` 复制为 `.local/teacher.json`，把其中 `output`、`export` 改为新的目录，再指定所选清单：

```bash
# 也可将 --pool 换为其他现成的 Train 采集清单。
python -m robustops.data.teacher_collection \
  --config .local/teacher.json --pool .local/sft_pool.json --stage all
```

使用自采数据训练出的 SFT 模型启动 RL 时，在本地 RL 配置中将 `model.path` 指向导出的 Hugging Face 模型目录，并通过 `data` 字段绑定对应任务池、候选标签、SFT 记录和故障暴露来源。启动器检查模型文件和数据的一致性，不要求固定为 692 条示范或第 174 步检查点。具体字段见[训练指南](../recipes/training/README.zh-CN.md)。

### 轨迹转 SFT 数据

[teacher_sft](../src/robustops/data/teacher_sft.py) 读取带教师指导的 DeepSeek 推理目录中的 `config.json`、`attempt.json` 和 `calls/*.request.json`，统一完成去指导、监督掩码 处理、上下文验收和训练数据导出。`source` 可指向单次推理目录，或包含温度 0/0.6 两轮的采集目录；同一条件优先保留首轮通过验收的轨迹。

```bash
# 修改副本中的 source、output 和 manifest，处理已有推理产物。
mkdir -p .local
cp recipes/data/teacher_sft.json .local/teacher_sft.json
python -m robustops.data.teacher_sft --config .local/teacher_sft.json
```

`output` 使用新目录；默认配置导出到 `runs/teacher_sft/train/`：

| 文件 | 内容 |
|---|---|
| `records.jsonl` | 去掉教师指导后的完整对话、工具与逐助手决策的监督策略 |
| `masks.jsonl` | 每条轨迹的监督策略、Base tokenizer 下参与 loss 的 token 区间与上下文验收结果 |
| `mask_changes.jsonl` | 被屏蔽或调整的决策、规则和原因 |
| `train.parquet` | 可直接绑定到 SFT 配置的训练数据，包含对话、工具和监督策略 |
| `provenance.jsonl` | 记录与原始推理目录、条件和温度的对应关系 |

自动 mask 针对被拒绝且违反暴露 schema 的调用。可通过 `reviewed_annotations` 指定已审查的叙述标注；模块保留调用与恢复历史，按标注屏蔽不适合模仿的解释。训练直接读取 Parquet 中的监督策略，以相同 tokenizer 重建 loss mask，将每条轨迹的监督权重在保留的助手决策之间均分，再在各决策参与监督的 token 之间均分。具体行为、三种监督目标及保留规则见 SFT loss mask 说明（[SFT loss masks](https://www.github.com/haohuiliang/RobustOps/blob/main/data/sft/README.zh-CN.md#user-content-教师导出的-loss-mask-与目标行为)）。

### Val 快速监控数据

Val 使用同一处理模块：在配置副本中将 `partition` 设为 `validation`，用 `manifest` 指定 Val 任务清单，`source` 指向对应的带指导推理目录，`output` 指定新目录，执行上面的处理命令即可。导出 `validation/validation.parquet` 后，自行筛选一小部分记录，用于快速监控 Val 集表现，将文件绑定到本地 SFT 配置的 `data.validation_parquet`，并更新 `data.expected_validation_records`。

## 采集 SFT Val Pass@4 轨迹

[`sft_val_pass4.json`](../recipes/inference/sft_val_pass4.json) 对 127 个案例的 clean/fault 各采样四次，共 1,016 条，温度 0.6。每次从初始状态完整执行，使用独立 seed，故障类型和重复次数上限均来自同一份 Base 固定清单。Pass@4 按四次中至少一次任务成功统计。

```bash
# 执行 SFT Val Pass@4。
bash run.sh inference --config recipes/inference/sft_val_pass4.json
# CPU 下查看计划，不启动模型或 MCP。
bash run.sh inference --config recipes/inference/sft_val_pass4.json --gpus 1 --dry-run
```

模型读取 `models/sft`，默认输出到 `runs/sft_val_pass4/`。`--gpus`、`--output` 和 `--base-fixed-mechanisms` 的覆盖方式与固定评估一致。默认 `resume: true`，再次执行相同命令只完成剩余采样。结果保存在根目录 `summary.json`，逐案例四次结果位于 `sft/summary.json`。

这些 Val 轨迹用于评估采样后的任务完成情况，不进入 SFT 或 RL 训练。

SFT 格式（[SFT format](sft/README.zh-CN.md)） · [执行轨迹](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories) · [上游归属与许可](../THIRD_PARTY_NOTICES.md)
