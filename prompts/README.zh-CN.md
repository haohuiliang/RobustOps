# LLM 提示词与固定案例

[English](README.md) · [简体中文](README.zh-CN.md)

这三份提示词分别用于 SFT 教师采集、故障选择和检验依赖图生成，统一使用 **DeepSeek-V4.1-Flash**。任务自带的 system/user prompt 来自任务资产，脚本直接读取这里的正文；评估不会向智能体附加这些指导。

| Prompt | 推理输入 | 推理输出 | 读取入口 |
|---|---|---|---|
| SFT 教师指导提示词（[teacher.md](teacher.md)） | 原始任务与策略、工具、逐步累积的 assistant/tool 历史。 | 实际执行的工具调用与最终回复；通过验收的轨迹去除指导、生成监督掩码后成为 SFT 数据。 | `teacher_reproduction` 与 `case_inference`，通过 `teacher_guidance()` 读取。 |
| 故障选择提示词（[fault_selector.md](fault_selector.md)） | 任务与策略、工具 schema、服务器路由、verifier 元数据及原生返回形状观测。 | 覆盖每个工具的 JSON：角色、理由、候选故障、恢复工具及安全重放依据。 | `fault_candidates` / `fault_dataset`。 |
| 检验依赖图提示词（[verifier_graph.md](verifier_graph.md)） | 任务与策略、带索引的 SQL 检查及期望值/比较方式、相关数据库 schema；不提供种子数据行或解题轨迹。 | 保留全部 verifier 节点的 JSON，包含前置依赖索引、证据和不确定性。 | `scripts/build_verifier_graphs.py`。 |

## SFT 引导

可直接使用提供的 SFT 数据（[SFT data](../data/sft/README.zh-CN.md)），也可按采集 CLI（[Collection CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成或采集)）自行采集。带指导的推理产物由统一的轨迹转 SFT 模块（[teacher_sft.py](../src/robustops/data/teacher_sft.py)）移除指导、处理监督掩码 并导出训练数据；处理 CLI（[Processing CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-轨迹转-sft-数据)）。

早期尝试只在采集后筛选、改写和屏蔽，仍有漏判和误删，也无法补出未执行的正确路径，因此后来在采集时增加通用指导。指导用于采集适合 SFT 的执行示范，减少虚构结果、叙述与动作矛盾和遗漏剩余工作。它保留原任务范围与依赖，允许合理推断与探索，并继续可独立执行的工作，不提供逐题规则、标准答案或隐藏故障信息。恢复规则按证据触发：明确未执行的调用才有限重试；写入状态未知时先查状态，避免重复副作用；缺失或冲突证据才重新查询。可靠回执无需额外重复检查。

更强的变体未证明稳定的监督质量收益，无指导对照有时反而完成更多任务。因此保留通用指导、检查实际决策，避免不断追加逐题例外；取舍见 [教师指导的取舍](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-为什么增加教师指导)。当前正文与教师采集所用文本一致。`consistent_narration` 读取正文，`none` 返回空字符串；只有 `teacher_data` 模式附加指导。采集按去掉指导后的学生可见上下文计数；导出时去掉教师指导，并检查完整样本是否在 32K 内，评估始终不加指导。上下文对齐方式见统一执行方式（[Execution](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-统一执行方式)）。

### 提示词编写原则

- 围绕监督价值约束可观察行为：说明简短，执行与叙述一致，完成声明有可见依据，不把部分完成说成全部完成。
- 按原请求的对象、范围与依赖行动，允许合理推断和探索；不虚构操作，不追加原任务没有的完成要求。
- 故障恢复按证据触发：明确未执行时有限重试，写入状态未知时先确认，读取不完整时补查；保留已完成进展并继续剩余工作。
- 可靠回执无需反复核验，仍有可执行工作时不提前结束；不强制固定工具顺序或统一重试次数。
- 使用通用规则，不提供逐题答案、verifier 内容或隐藏故障信息；导出学生数据和运行教师评估时移除指导。

[data/sft/](../data/sft/) 提供可直接使用的教师示范与监督掩码。DeepSeek-V4.1-Flash 根据原始任务、策略和执行中的工具返回，在教师指导下采集轨迹，再经任务成功、上下文验收与监督处理后导出。额外教师指导在导出时移除，学生输入保留原始任务与真实执行历史。

## 故障选择

故障选择器的输入包括原始任务与策略、工具的 schema 和服务器路由、verifier 元数据，以及从正常工具响应中提取的原生返回形状。它据此判断哪些故障可能适用于任务中的合法操作。

选择器对每个工具初筛七个标签，列出可能适用的机制、理由与恢复路径，缩小候选范围。初筛依据真实工具和支持的语义，例如 partial read 需要原生 count 证据，未知写入需要可行的读回确认或安全重放依据。不虚构接口，也不为凑覆盖率增加标签。

最终能否注入，由运行时注入器根据实际参数、返回、历史和剩余重复次数判断。筛选条件失效（`ignored_filter`）不参与本项目 SFT/RL 训练，Val/Test 可用它检验对未见故障机制的泛化能力。提供的返回形状观测来自此前正常响应，不是额外故障标签。

选择器扫描结果在 [data/fault_selector/](../data/fault_selector/)，包含 Train、Val、Test。每个工具的标签限定后续可尝试的故障范围；调度器再选机制和预算，注入器检查实际调用。Test 工具标签也保存在固定评估记录（[policy.json](../artifacts/test_fault_policy/policy.json)），与 [test.jsonl](../data/fault_selector/test.jsonl) 一致；它不是最终机制分配。以下命令生成自己的标签，后续绑定见数据指南（[Data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成故障标签)）：

```bash
# 使用 DeepSeek-V4.1-Flash 重新生成候选标签，写入 runs/fault_selector/dataset。
bash run.sh inference --config recipes/inference/fault_selector.json
```

### 提示词编写原则

- 依据任务原意、工具 schema、真实服务器路由与原生返回形状，判断合法相关用途；不虚构字段、接口或能力，不按工具名猜测语义。
- 接受合理替代路径与可通过工具获取的 ID。没有观察到返回形状、尚无历史或当前结果为空，不自动等同于能力不存在；但不能把不确定性当作肯定证据。
- 恢复路径必须可执行：提交未知时说明可用的读回工具、丢失回执后如何取得参数，或给出有语义依据的安全重放方式；父对象元数据不能代替子对象效果。
- 每个工具逐项独立检查所有故障，先写证据理由再给一致的标签；允许空标签，不按案例打补丁或为覆盖率凑标签。
- 标签只限定候选范围。实际参数、返回、计数、历史与状态由注入器核验；verifier 只辅助理解任务，其隐藏值不作为智能体已知信息。

[data/fault_selector/](../data/fault_selector/) 提供可直接使用的 Train、Val、Test 候选标签，由 DeepSeek-V4.1-Flash 根据上述输入离线辅助生成。标签供调度器和注入器使用，不作为额外提示加入执行模型上下文。

## 检验依赖图生成

检验依赖图生成的输入包括原始任务与策略、带索引的 SQL 检查及其比较符和期望值，以及相关数据库 schema；不提供初始数据行或解题轨迹。

检验依赖图生成针对部分进度的误导：原任务要求的前置上下文未满足时，下游检查不应获得信用。只增加检查之间的依赖，不新写 SQL、不追加条件、不改变成功标签。独立编辑、否定结果和清理操作保持独立，除非检查所衡量的关系确实要求前置依赖。不确定的边省略，空图也有效。

早期尝试过细化 SQL 检查，采用的提示词保留节点及等权关系。先按字面理解每个布尔条件，再判断其他检查是否补充了缺失的任务上下文、功能资格或明确完成声明对应的事实。共用表、操作先后或泛泛的完成消息，不足以构成依赖证据。

设计需要平衡两种风险：漏掉必要上下文，会奖励错误主体上的操作；连接过多，则会压掉有效的独立进展。单对象检查确实补充必要父对象或角色时，即使捆绑了其他字段也可作为锚点，但必须披露这些字段；不能只为要求完整配置、唯一数量或下游查询已确认的存在性而连边。每条边都要有证据，传递依赖也不能夹带无关工作。缺少所需的更细检查时报告限制，不自行补造。

### 提示词编写原则

- 按 SQL、比较符和期望值理解条件，保留全部节点及等权关系；不新增 SQL、检查条件或更高的成功要求。
- 只连接必要上下文：被测关系缺少的父对象、受益人、角色资格，或明确完成声明所对应的事实。共用表、操作顺序或泛泛的完成消息不足以连边。
- 保留独立字段修改、数量、否定结果与清理的信用；已有查询确认的存在性无需再次门控。粗检查确有必要上下文作用时可作为锚点，但须披露捆绑的独立字段。
- 每条边说明证据，并检查传递依赖是否夹带无关工作。证据不足则省略，允许空图并报告缺失绑定；不声称图能修复含糊的 SQL join。

程序验证节点完整、无环和工具配置下的顺序映射，再检查全成功条件等价。布尔结果为 `v_i`，门控结果为 `g_i = v_i AND all(g_j for prerequisites j)`，其均值用于进度信用；最终任务成功仍要求检查全部通过。图不能修复含糊的 SQL join，也不能补出任务成功标准中不存在的更细检查。

提供的 Train/Val 检验依赖图（[verifier_graph](../data/verifier_graph/)）可直接使用，也可自行生成：

```bash
# 使用 DeepSeek-V4.1-Flash 重新生成 Train/Val 依赖图，写入 runs/verifier_graph/dataset。
python scripts/build_verifier_graphs.py --config recipes/data/verifier_graph.json
```

生成时在运行目录保存提示词及请求与响应记录。检验依赖图生成请求用 `--phase run` 继续，再用 `--phase export` 导出；修改提示词后需要新的运行目录。自行生成的标签和图写入 `runs/`，提供的数据保持不变；配置绑定见数据指南（[Data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成或采集)）。

API 密钥设置见[教师 API 配置](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-教师-api-配置)。

[data/verifier_graph/](../data/verifier_graph/) 提供可直接使用的图，由 DeepSeek-V4.1-Flash 根据任务、检查和 schema 离线辅助生成。SQL、期望值和图元数据不进入执行模型的上下文。

## 运行案例

本地模型与 API 教师使用同一个配置驱动的推理入口：

```bash
# 运行固定 SFT 案例的 clean/fault 评估。
bash run.sh inference --config recipes/inference/cases.json
# 使用 BF16 RL 运行同一固定案例。
bash run.sh inference --config recipes/inference/rl_cases.json
# 在评估模式下运行教师模型，不附加教师指导。
bash run.sh inference --config recipes/inference/teacher_cases.json
# 运行教师模型，启用训练数据指导与导出检查。
bash run.sh inference --config recipes/inference/teacher_data_cases.json
```

| 配置 | 模型 | 模式 | 教师指导 |
|---|---|---|---|
| `cases.json` | SFT | `evaluation` | 无 |
| `rl_cases.json` | BF16 RL | `evaluation` | 无 |
| `teacher_cases.json` | API 教师 | `evaluation` | 无 |
| `teacher_data_cases.json` | API 教师 | `teacher_data` | `teacher.md` |

`data/cases/train.json` 固定任务、工具、初始状态、seed 与故障。`data/cases/test.json` 提供筛选条件失效（`ignored_filter`）的泛化评估示例，可在本地评估配置中绑定。教师数据模式默认使用提供的 80% Train 清单，可通过 `teacher.pool` 选择其他 Train 采集清单；仍排除 Val/Test 和 `ignored_filter`。选范围与生成比例的 CLI 见采集清单说明（[Collection manifests](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-选择教师采集清单)）。

clean/fault 从同一数据库开始，故障只在符合资格的调用施加；`fixed_call_indices` 可进一步限定位置。每个条件运行一次，不执行完整 Test 的补采。统一 temperature 0、50 轮及 Base tokenizer 统计的 32K 上下文 / 2K 输出预算。

教师数据模式去除指导，检查成功候选是否符合学生上下文后导出，候选仍需数据审查；评估模式只生成评估结果。这使同一入口能生成案例，同时保持 Test 与训练隔离。

API 推理沿用上述密钥配置。[运行环境准备](../backends/enterpriseops/README.zh-CN.md) · 教师数据采集 CLI（[Teacher collection CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)）
