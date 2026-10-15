# 配置入口

[English](README.md) · [简体中文](README.zh-CN.md)

先按主 [README](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-1-安装环境) 完成环境安装、模型下载和业务环境准备，再选择评估、教师数据采集或训练任务。本文说明各配置的用途、故障策略及自行生成数据后的绑定方式。所有命令均在仓库根目录执行；下表中的配置路径相对于 `recipes/`。

默认配置直接读取仓库提供的故障候选标签、检验依赖图、SFT 示范及 Val/Test 固定故障清单。使用这些数据时，无需先运行生成或采集流程。

## 评估、教师数据与训练

| 操作 | 配置 | 用途 |
|---|---|---|
| Base 评估 | `inference/base_test.json`、`inference/base_val.json` | 在固定 Test/Val 上运行无注入与故障条件的配对评估。 |
| SFT 评估 | `inference/sft_test.json`、`inference/sft_val.json` | 使用 SFT 模型，在相同任务和固定故障清单下评估。 |
| RL 评估 | `inference/rl_test.json`、`inference/rl_val.json` | 使用 RL 模型，在相同任务和固定故障清单下评估。 |
| 教师评估 | `inference/teacher_test.json`、`inference/teacher_val.json` | 通过 API 评估教师参考模型，不追加教师指导。 |
| SFT Val Pass@4 | `inference/sft_val_pass4.json` | 温度 0.6；每个 Val 案例在无注入、故障条件下各独立采样四次。 |
| 单任务案例 | `inference/cases.json`、`inference/rl_cases.json`、`inference/teacher_cases.json` | 运行本地模型或教师 API 案例，见[提示词与案例](../prompts/README.zh-CN.md)。 |
| 教师示范案例 | `inference/teacher_data_cases.json` | 运行带教师指导的采集案例。 |
| 教师数据采集 | `data/teacher.json` | 在所选 Train 任务池中采集、验收并导出 SFT 示范；用 `pool` 或 `--pool` 指定采集清单。 |
| 已有轨迹转 SFT 数据 | `data/teacher_sft.json` | 从带指导的教师推理记录中移除指导，生成监督掩码并导出 JSONL/Parquet。 |
| SFT 训练 | `training/sft.json` | 从 Base 开始，使用提供的 692 条 Train 示范训练；原实验采用第 2 轮 / 第 174 步检查点。 |
| RL 训练 | `training/rl.json` | 从 SFT 检查点开始，在 Train 任务上训练，默认目标为 120 步。 |

训练和模型推理使用 `run.sh`，教师数据采集与转换使用相应 Python 模块：

```bash
# 使用下载的 SFT 模型，在固定 Test 上评估。
bash run.sh inference --config recipes/inference/sft_test.json
# 可选：自行采集教师示范，并完成数据导出。
python -m robustops.data.teacher_collection --config recipes/data/teacher.json --stage all
# 使用提供的示范，从 Base 开始训练 SFT。
bash run.sh train --config recipes/training/sft.json
```

教师 API 的密钥配置见[下文](#教师-api-配置)。采集范围、失败案例补采和数据导出见[数据指南](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)；GPU、检查点、续训及重训后的评估见[训练指南](training/README.zh-CN.md)。

## 生成数据与汇总结果

| 操作 | 配置 | 用途 |
|---|---|---|
| 生成故障候选标签 | `inference/fault_selector.json`，内部读取 `data/fault_selector/all.json` | 扫描 Train/Val/Test，输出可能适用的工具×故障组合。 |
| 生成检验依赖图 | `data/verifier_graph.json` | 生成 Train/Val 的检验依赖图，供 RL 计算部分进展奖励。 |
| 运行 Base 并分配故障 | `inference/base_fault_assignment_test.json`、`inference/base_fault_assignment_val.json` | 先运行无注入轨迹，再从实际合格机会中选择故障机制和次数上限，执行故障续行。 |
| 保存固定故障清单 | `data/freeze_test.json`、`data/freeze_val.json` | 将本次 Base 运行选定的机制和次数上限保存为固定清单，供各模型共享。 |
| 汇总评估结果 | `data/summarize.json`、`data/summarize_teacher.json` | 分别汇总 Base/SFT/RL，或加入教师模型进行四模型比较。 |

故障标签和依赖图的生成命令、输出及后续使用方式见[数据指南](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成或采集)。下文说明如何重新生成 Val/Test 的固定故障清单，以及如何接入新的选择器标签。

## 教师 API 配置

教师 API 密钥任选一种方式配置：

- 文件：在提供的 [`.env.deepseek`](../.env.deepseek) 中填写 `DEEPSEEK_API_KEY`。
- 环境变量：在当前终端执行 `export DEEPSEEK_API_KEY='你的密钥'`。

教师 Val/Test、API 案例及三类教师产物的生成或采集均支持这两种方式，同时配置时环境变量优先。单任务的本地/API 案例见提示词与案例（[Prompts and examples](../prompts/README.zh-CN.md)）。

## 故障注入策略

RL、教师采集与推理共用 [strategy.json](enterpriseops/faults/strategy.json)。选择器标签只限定需要检查的工具与机制；注入器根据实际调用参数、返回值和业务状态判断是否满足注入条件。候选标签可能不适用于某次调用，也可能整条轨迹都没有合格机会。

**从无注入轨迹的保存点续行。** 生成新 Base Val/Test 分配、采集教师首轮数据和 RL 训练采用这条路径：无注入（`clean`）阶段记录合格机会，并保存调用前的业务状态与对话。调度器随后选择机制、重复次数上限和注入位置，恢复该位置的状态与对话，执行调用并开始故障续行；保存点之后的 clean 回答不用于故障分支。

- Base 分配和教师首轮采集先完成本批同温度下的全部 clean 案例，再按案例顺序分配故障，兼顾故障类型和业务域的覆盖，遇到并列候选时用固定 seed 选择。注入从所选机制最早的合格位置开始；没有合格机会的案例保持零注入。
- RL 使用当前策略的 clean 轨迹，每条轨迹完成后即可参与故障组调度，无需等待整批 clean 完成。所选位置还必须留有故障续行所需的上下文和轮数预算；没有合格保存点的轨迹不能提供有效故障组。

**使用固定清单进行评估。** 各模型从任务初始状态开始执行，使用清单指定的机制和次数上限。遇到该机制第一次合格的调用时开始注入，之后在合格调用处继续注入，直至达到上限或轨迹结束。相同清单约束的是机制和上限，各模型的实际调用路径与命中次数仍可能不同；整条轨迹没有合格调用时，实际注入次数为零。教师定向补采的 `eligible_now` 路径也在线检查，但在当前调用合格的候选中按优先级选择机制，选中时才抽上限，并非预先固定唯一机制。

**重复次数上限。** 动态选择机制后，每个计划只抽一次上限：暂时拒绝（`transient_unexecuted`）从 1、2、4 中各以 1/3 概率抽取；不确定提交（`unknown_commit`、`unknown_commit_create`）与筛选条件失效（`ignored_filter`）从 1、2 中各以 1/2 概率抽取；集合缺项、响应截断与响应串包从 1、4 中各以 1/2 概率抽取。抽取不依赖任务成败，也不要求小批次凑齐这些比例。后续合格调用消耗剩余次数；上限不保证最终实际命中次数。

**RL 注入位置。** 默认 `dose_window_uniform` 按所选机制的合格位置数选择窗口。设通过续行预算检查的位置共有 `N` 个，次数上限为 `B`，则从最早的 `ceil(N / B)` 个位置中均匀随机选择，seed 固定时可复现。因此上限为 1 时可选全部位置，为 2 时可选前半段，为 4 时可选前四分之一，向上取整且至少保留一个。例如 `N = 10` 时，三个窗口分别包含 10、5、3 个位置。这里的窗口按合格调用位置计数；token 长度和最大轮数用于筛除没有续行空间的位置。

新 Base 运行记录选定的机制和次数上限，随后可将这些结果保存为各模型共用的固定清单。已发布的固定清单保留原先记录的取值，后续评估不重新抽样。`ignored_filter` 整个机制族不参与本项目 SFT/RL，用于评估对未见故障机制的泛化能力。教师采集入口见[数据指南](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)。

固定故障评估可对运行有效、协议正常但没有实际命中的案例换 seed 补采。任务失败或未耗尽次数上限本身不触发补采。基础设施及协议错误使用独立的重试预算；动态 Base 分配和教师保存点续行不使用这类零命中补采。

实现见[共享策略与分配](../src/robustops/evaluation/fault_schedule.py)、[保存点分支](../src/robustops/evaluation/clean_fault_branch.py)、[在线固定机制](../src/robustops/evaluation/selected_fault.py)和 [RL 位置窗口](../src/robustops/training/rl_fault_runtime.py)。

## 重新生成并使用故障配置

默认评估直接读取已提供的 [Test](../artifacts/base_test_fixed_mechanisms.json) 或 [Val](../artifacts/base_val_fixed_mechanisms.json) 固定清单。如需根据一次新的 Base 运行重新分配故障，先运行分配入口，再将本次选定的机制和次数上限保存为固定清单。以下命令复用提供的候选标签。

```bash
# Test：运行 Base，由调度器生成本次故障分配。
bash run.sh inference --config recipes/inference/base_fault_assignment_test.json \
  --output runs/my_base_test
# 读取上一步结果，生成供各模型共享的固定清单。
python -m robustops.evaluation.reproduction \
  --action freeze --config recipes/data/freeze_test.json \
  --base-run runs/my_base_test \
  --output runs/custom_faults/my_test.json
```

`freeze` 读取已完成的 Base 运行，不再调用模型。若案例没有选定合格机制，或运行记录与任务、次数上限不一致，命令会报错，不会为缺失案例编造机制。

后续评估用 `--base-fixed-mechanisms` 指向新清单，并用 `--output` 指定新的输出目录。参与比较的模型应在同一份新清单下重新运行配对评估：

```bash
# 使用新故障配置分别评估 SFT 和 RL。
bash run.sh inference --config recipes/inference/sft_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_sft_test
bash run.sh inference --config recipes/inference/rl_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_rl_test
# 在同一固定清单下重新运行 Base 的无注入/故障配对评估。
bash run.sh inference --config recipes/inference/base_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_base_comparison
# 可选：教师参考也使用同一新配置。
bash run.sh inference --config recipes/inference/teacher_test.json \
  --base-fixed-mechanisms runs/custom_faults/my_test.json --output runs/my_teacher_test
```

省略这些参数时仍使用配置文件中的默认路径。参数只覆盖本次运行，不改写 JSON 配置或提供的固定清单。每套新故障配置使用新的输出目录；需要对比时，参与的模型都在同一新清单下重新评估。

Val 使用对应的分配、冻结及评估入口：

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
# Base Val 对照同样通过参数选择新清单。
bash run.sh inference --config recipes/inference/base_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_base_val_comparison
# 可选：教师 Val 也使用这份新清单。
bash run.sh inference --config recipes/inference/teacher_val.json \
  --base-fixed-mechanisms runs/custom_faults/my_val.json --output runs/my_teacher_val
```

## 将新标签用于评估

如果只是重新运行 Base 并分配故障，可以复用提供的候选标签，直接执行上一节。若重新运行选择器，则需要区分两步：先将选择器结果整理为评估读取的 `policy.json`，固定任务输入和候选标签；再运行 Base，生成并保存每个案例最终使用的故障机制及次数上限。选择器输出的 JSONL 不能直接作为 `selector_policy`。

以 Test 为例，先准备选择器配置副本：

```bash
mkdir -p .local
cp recipes/data/fault_selector/test.json .local/test_selector.json
```

在副本中设置以下字段，其余字段沿用原配置：

| 字段 | 示例值 | 用途 |
|---|---|---|
| `output` | `runs/my_test_selector` | 保存本批选择器请求和结果。 |
| `schedule` | `recipes/enterpriseops/faults/test.json` | 使用 Test 对应的故障策略配置。 |
| `tokenizer` | `models/base/Arctic-MidTool-RL-4B` | 指定后续评估使用的 tokenizer。 |
| `policy_output` | `runs/my_test_selector/policy.json` | 指定评估记录的输出路径。 |

```bash
# 调用选择器；再次执行时复用已有有效结果。
python -m robustops.data.fault_candidates --config .local/test_selector.json --phase select
# 将已保存的结果整理为 policy.json，不再调用 API，也不预先抽取次数上限。
python -m robustops.data.fault_candidates --config .local/test_selector.json --phase freeze
```

然后复制需要使用的 Base 分配、机制冻结及模型评估配置，将各副本中的 `selector_policy` 统一改为 `runs/my_test_selector/policy.json`。Base 分配使用对应分区的 `schedule`；机制冻结用 `--base-run` 指向这次 Base 输出，优先读取该目录保存的 `candidate_budgets.json`。按上一节依次完成分配、冻结和评估，并将命令中的 `--config` 改为这些副本。

Val 对应使用 `recipes/data/fault_selector/val.json` 及各 Val 配置。每一组配置都应绑定同一分区的候选标签、Base 分配结果和固定机制清单。

## 汇总评估结果

```bash
# 汇总 Base、SFT、RL 的固定 Test 评估。
python -m robustops.evaluation.reproduction --action summarize --config recipes/data/summarize.json
# 可选：加入教师参考模型。
python -m robustops.evaluation.reproduction --action summarize --config recipes/data/summarize_teacher.json
```

默认三模型汇总读取 `runs/base_test`、`runs/sft_test`、`runs/rl_test`；四模型汇总再读取 `runs/teacher_fixed_test`。结果分别写入 `runs/comparison/summary.json` 和 `runs/comparison/summary_with_teacher.json`。使用前文自定义目录时，复制汇总配置，将 `runs` 中各项的 `path` 改为实际评估目录，并设置新的 `output`。

## SFT Val Pass@4

`inference/sft_val_pass4.json` 对 127 个 Val 案例的 clean/fault 条件各独立采样四次，共 1,016 条轨迹，温度为 0.6。各次运行使用独立 seed，故障机制和次数上限来自同一份 Base 固定清单。运行、续跑及输出说明见[采集指南](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-采集-sft-val-pass4-轨迹)；这些轨迹用于评估，不进入 SFT/RL 训练。

## 本地配置

将需要修改的配置复制到 `.local/`，再调整模型、数据和输出路径，以及 batch、学习率或步数。自行采集的数据通过配置绑定到训练流程，具体字段见[数据指南](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成或采集)和[训练指南](training/README.zh-CN.md)。SFT/RL 的 Val loss 用于诊断，不参与参数更新。

提供的 Base/SFT/RL 固定 Val/Test 评估默认使用一张 32 GB GPU，以 BF16 计算，每卡并发数 `concurrency_per_gpu` 为 `2`。建议最低配置为单卡 16 GB，此时在评估配置中将 `concurrency_per_gpu` 设为 `1`。`hardware.gpus` 设置 GPU 数量，`--gpus` 可覆盖该值，`CUDA_VISIBLE_DEVICES` 选择具体设备。教师评估通过 API 执行。

训练省略 `--gpus` 时使用全部可见显卡；`hardware.recorded` 只记录原实验硬件。训练和模型评估可加 `--dry-run` 查看计划；在没有 GPU 的机器上，用 `--gpus N --dry-run` 明确待检查的 GPU 数量。故障选择器入口的 dry-run 会准备请求文件，但不调用 API。

`support/` 保存训练与验证的配套配置；`enterpriseops/` 保存各业务域、工具配置和故障策略。已发布模型的下载位置见主 README 的[模型准备](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-2-准备模型资产)：默认 RL 评估从 `models/rl` 读取 BF16 权重，原始 FP32 导出放在 `models/rl_fp32`。训练检查点的保存、恢复和 Hugging Face 模型导出见[训练指南](training/README.zh-CN.md)。
