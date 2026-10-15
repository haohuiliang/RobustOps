# SFT 数据

[English](README.md) · [简体中文](README.zh-CN.md)

这些可直接使用的记录随 GitHub 仓库提供，训练配置直接读取，无需再从 Hugging Face 下载训练数据。

| 划分 | 记录数 | 文件与用途 |
|---|---:|---|
| Train | 692 | `train/records.jsonl`、`train/train.parquet`、`train/provenance.jsonl` 和 `train/fault_exposure.json`；用于监督训练 |
| Validation | 15 | `validation/records.jsonl`、`validation/validation.parquet` 和 `validation/provenance.jsonl`；用于快速监控 Val 集表现，不参与梯度更新 |

这份数据采自按任务 ID 选取的 80% 可运行 Train 任务池；自行采集可选择其他 Train 清单或比例。有效示范覆盖以完整 **1,025 个可运行 Train 任务×工具配置**为分母，分别统计 clean 和 fault：

| 条件 | 记录数 / 不同组合数 | 示范覆盖率 |
|---|---:|---:|
| Clean | 357 / 357 | **357/1,025（34.83%）** |
| Fault | 335 / 335 | **335/1,025（32.68%）** |

289 个组合同时具有两种条件，去重后的联合覆盖为 **403/1,025 个组合（39.32%）**。

JSONL 与 Parquet 对应相同样本，包含任务、对话、工具和监督掩码。通过提供的配置使用 `DecisionSFTDataset`，保留选择性决策监督和轨迹权重。已识别的无效决策仍作为上下文，但屏蔽其目标 loss；合法调用遇到暂态故障不自动被屏蔽，有效的故障遭遇及后续恢复仍可学习。

```python
from datasets import load_dataset

records = load_dataset("parquet", data_files={
    "train": "data/sft/train/train.parquet",
    "validation": "data/sft/validation/validation.parquet",
})
```

教师来源通过 `record_id` 关联；`provenance.jsonl` 指向可选轨迹包（[RobustOps-Trajectories](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories)）中的带指导教师原始轨迹。处理后的训练记录、监督策略及来源信息随仓库提供。教师数据采集（[Teacher collection](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-教师数据采集)）说明采集范围与成功转化情况。

自行采集见数据指南（[Data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)）。统一的轨迹转 SFT 模块（[teacher_sft.py](../../src/robustops/data/teacher_sft.py)）把带指导的推理产物处理为完整对话、监督掩码、修改原因和 Parquet；处理 CLI 与产物说明（[SFT processing](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-轨迹转-sft-数据)）。Val 使用相同模块，筛选一小部分记录用于快速监控 Val 集表现，见 Val 数据处理说明（[Val diagnostics](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-val-快速监控数据)）。

采集预算按去掉教师指导后的学生可见上下文计算，导出保留实际执行历史，完整样本验收上限为 32K；详见统一执行方式（[Execution](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-统一执行方式)）。

## 教师导出的 loss mask 与目标行为

导出先确认任务成功、执行有效、环境清理成功及学生上下文不超过 32K，再决定各助手决策如何参与监督。教师额外指导从输入中移除；实际调用、参数、工具回执和助手历史保留。loss mask 将指定 token 的监督损失置零，并不把它们变成负向训练目标。

| 行为或内容 | 判定依据与处理 | 希望保留的学习目标 |
|---|---|---|
| 使用未暴露工具，或参数违反已暴露 schema，且调用被拒绝 | 自动核对工具名、参数约束与该次错误回执，将整个助手决策设为 `masked`；首次发生也可屏蔽 | 学习按工具定义调用，以及后续有依据的纠正 |
| 动作可保留，但附带叙述有已审查的问题 | 通过 `reviewed_annotations` 指定 `tool_call_only`，例如叙述声称完成了实际并未执行的操作；标注必须匹配原助手消息 | 学习真实工具动作，不模仿错误叙述 |
| `tool_call_only` 决策中工具调用前的解释或规划 | 作为输入上下文保留，不计入目标 loss | 保留决定动作所需的上下文，监督工具调用本身 |
| 合法调用遇到暂态错误、注入故障或不符合预期的业务结果 | 不因返回错误就自动屏蔽；自动规则需要同时满足工具/schema 违规和调用被拒绝 | 学习合理尝试、利用错误反馈及后续恢复 |
| 其余保留的有效助手决策 | 使用 `full`，包括有依据的叙述、工具动作和完成回答 | 学习有效执行及有据结束 |
| 系统消息、用户消息、工具返回 | 始终作为上下文，不是助手预测的监督目标 | 从任务、策略与回执中选择后续动作 |

三个监督目标的实际范围是：`full` 覆盖该助手决策的生成 token；`masked` 不监督该决策；`tool_call_only` 从 `<tool_call>` 开始到该助手消息结束，调用前正文不参与 loss。并非所有规划都被屏蔽，只有明确设为仅作上下文的部分如此处理。

自动导出不执行完整的语义审查，也不会仅凭任务最终成功就证明每句话都正确。不实叙述的处理依赖提供的已审查标注；默认 `reviewed_annotations: null` 不会自动生成这些标注。比如，参数不在工具公布的枚举中且调用被拒绝，会触发自动 mask；参数合法但服务暂时不可用则不会。若叙述声称“两个对象都已更新”，实际只调用了一次工具，需要审查标注才能屏蔽这段叙述并保留可学习的调用。

每条轨迹的总监督权重先在未被完全屏蔽的助手决策间均分，再在各决策参与监督的 token 间均分。例如总权重为 1、两个决策分别有 20 和 100 个监督 token 时，两步各占 0.5，每个 token 分别占 0.025 和 0.005。较长的决策不会因此获得更大的总权重。

可在导出的 `records.jsonl` 中查看 `assistant_supervision`，在 `masks.jsonl` 中查看 token 区间，在 `mask_changes.jsonl` 中查看规则与原因。已提供数据的监督策略以记录本身为准。

实现：导出与标注处理（[teacher_sft.py](../../src/robustops/data/teacher_sft.py)） · 监督范围与权重（[selective_supervision.py](../../src/robustops/training/selective_supervision.py)） · 处理命令与产物（[SFT processing](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-轨迹转-sft-数据)）。RL 的行为处理见局部信用规则（[Local credit](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/training/README.zh-CN.md#user-content-局部信用的目标行为)），其负优势处理与 SFT 的零 loss mask 分别定义。
