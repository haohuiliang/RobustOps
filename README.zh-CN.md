<h1 align="center">RobustOps</h1>

<p align="center">
  <strong>通过监督微调与强化学习，提升 4B 企业工具智能体的任务完成与故障恢复能力。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python 3.12">
  <a href="https://huggingface.co/collections/haohuiliang/robustops-6ad051ea2e58c851faa61dd0"><img src="https://img.shields.io/badge/Hugging_Face-Models_%26_Data-yellow?logo=huggingface" alt="Hugging Face 模型与轨迹合集"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start">快速开始</a> · <a href="#models">模型与数据</a> · <a href="#training">训练</a> · <a href="#evaluation">评测</a> · <a href="#results">结果</a>
</p>

<p align="center">
  <strong>故障恢复训练 · MCP 企业任务 · 状态核验 · 开放模型与轨迹</strong>
</p>

---

RobustOps 是一个面向企业工具调用智能体的开源大语言模型后训练项目，结合监督微调（SFT）与智能体强化学习（RL），提升多步任务中的工具调用可靠性、智能体容错与故障恢复能力。更新可能已提交但确认丢失，读取可能不完整，返回内容也可能属于另一个请求。项目结合可执行的故障与恢复环境、教师监督数据和状态验证反馈，让智能体学习根据实际反馈调整决策、核对状态并继续完成任务。

基础任务来自 [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)，覆盖日历、邮件、云盘、客服、人事、IT 服务管理和团队协作。模型根据自然语言请求，通过模型上下文协议（MCP）调用工具，查询、更新和创建资源，完成单域或跨域的多步业务任务；最终由数据库状态检查（`verifier`）判定是否完成。

<a id="results"></a>

## 结果概览

本项目使用 DeepSeek-V4.1-Flash 作为教师模型。在固定测试集（Test）上，比较 Base → SFT → RL120 与教师模型的任务成功率：

| 评估口径 | Base | SFT | RL120 | DeepSeek-V4.1-Flash |
|---|---:|---:|---:|---:|
| 无注入（`clean`）：全部 130 项 | 15/130（11.54%） | 28/130（21.54%，+10.00 pp） | 40/130（30.77%，+9.23 pp） | 46/130（35.38%） |
| 故障（`fault`）：公共命中 116 项 | 9/116（7.76%） | 28/116（24.14%，+16.38 pp） | 39/116（33.62%，+9.48 pp） | 39/116（33.62%） |

Base → SFT → RL120 的提升均相对上一阶段，以百分点（pp）表示；RL 相较 Base 累计提升 **19.23 pp（无注入）和 25.86 pp（故障）**。上表的故障行使用补采后的四方公共实际命中集合。教师模型采用相同任务与评估设置，不追加教师指导。

## 方法概览

项目的主要工作包括：

- 故障环境：针对工具异常后结果不确定、任务难以继续的问题，设计可执行的故障注入与恢复回放流程，让模型在业务环境中学习核对状态、恢复任务。
- 教师数据：通过带指导的教师采集、成功验收与轨迹后处理，将教师解决任务的方法与路径整理为 SFT 示范，为学生策略提供冷启动。
- 训练策略：兼顾基础任务与故障恢复，让同组轨迹从相同故障后状态继续，减少起点差异，更直接地比较后续恢复决策。按组内学习信号动态筛选训练样本，并让零信号案例延迟回流，由更新后的策略重新采样。
- 学习信号与信用分配：为状态检查生成检验依赖图，按检查间的依赖计算有效进展，并结合任务成功与调用质量提供反馈。各项奖励分别归一化后再加权，减少尺度差异对预设权重的干扰；进一步将信用分配到具体动作片段，抑制坏调用与无进展重复读取，避免强化失败后的主动结束。 [GDPO](https://arxiv.org/abs/2601.05242)

训练基于 [veRL](https://github.com/verl-project/verl)，企业任务与环境来自 [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)。推理使用 2 × RTX 5090（每卡 32 GB）；SFT/RL 训练使用 2 × RTX PRO 6000 Blackwell（每卡 96 GB）。

<a id="quick-start"></a>

## 1. 安装环境

创建 `rops` 环境，再安装 [requirements.txt](requirements.txt) 中的依赖。后续命令均在仓库根目录执行。

```bash
# 获取源码。
git clone https://github.com/haohuiliang/RobustOps.git
cd RobustOps

# 创建环境并安装依赖。
conda create -n rops python=3.12 -y
conda activate rops
bash scripts/setup.sh
```

<a id="models"></a>

## 2. 准备模型资产

源码包含任务划分、故障候选、固定故障清单、检验依赖图和 SFT 数据。按需下载以下模型：

### Base

初始模型为 [Arctic-MidTool-RL-4B](https://huggingface.co/MidTool/Arctic-MidTool-RL-4B)，是一个已接受工具调用 SFT 与智能体 RL 训练的 Qwen3 系列 4B 模型。下文 Base 均指该检查点，即尚未经过本项目 SFT/RL 的模型。运行 Base、训练 SFT，以及教师评估所需的 tokenizer，均使用该模型。

```bash
# 下载初始 Base 模型及 tokenizer。
hf download MidTool/Arctic-MidTool-RL-4B --local-dir models/base/Arctic-MidTool-RL-4B
```

### SFT

[RobustOps-SFT-4B](https://huggingface.co/haohuiliang/RobustOps-SFT-4B) 为采用的第 2 轮 / 第 174 步模型。权重保存为 FP32，保留导出精度，可用于评估和继续训练，包括 RL 初始化；评估时以 BF16 加载。

```bash
# 下载 SFT 权重，用于推理或初始化 RL。
hf download haohuiliang/RobustOps-SFT-4B --local-dir models/sft
```

### RL BF16 — 推荐推理

[RobustOps-RL-4B-BF16](https://huggingface.co/haohuiliang/RobustOps-RL-4B-BF16) 为最终 step120 的 BF16 权重，推荐用于 RL 评估与推理。

```bash
# 下载建议用于 RL 推理的 BF16 权重。
hf download haohuiliang/RobustOps-RL-4B-BF16 --local-dir models/rl
```

### RL FP32 — 可选续训

[RobustOps-RL-4B](https://huggingface.co/haohuiliang/RobustOps-RL-4B) 保留导出权重的原始精度，方便继续训练。实际推理以 BF16 运行。

```bash
# 可选：下载 FP32 权重，用于继续训练。
hf download haohuiliang/RobustOps-RL-4B --local-dir models/rl_fp32
```

## 3. 准备数据集与业务环境

### 任务划分

Oracle 只暴露任务指定工具；plus5（`plus_5_tools`）在此基础上额外加入 5 个干扰工具，检验存在无关候选时的工具选择。

| 分组 | 任务数 | Oracle | plus5 | 案例总数 |
|---|---:|---:|---:|---:|
| Train | 518 | 518 | 507 | 1,025 |
| Val | 64 | 64 | 63 | 127 |
| Test | 66 | 66 | 64 | 130 |

案例按任务 × 工具配置统计。训练集（Train）、验证集（Val）与测试集（Test）先按任务 ID 划分，再展开工具配置，同一任务不会跨分组。任务清单（`manifest`）（[enterpriseops.jsonl](data/task_sets/enterpriseops.jsonl)） 固定上述任务及分组，总体说明（[enterpriseops.json](data/task_sets/enterpriseops.json)） 记录来源与划分规则。最终模型使用 Train 更新参数，Val 用于训练决策，Test 在训练过程中不可见，仅用于最终评估。

### 业务环境准备

企业任务、数据库和状态检查来自 [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym/tree/271f2c357f763376997dfd16807fcde2474ae41b)。准备其七域 MCP 运行环境：

```bash
# 准备七个业务域的 MCP 运行环境及项目适配。
bash backends/enterpriseops/prepare_udocker.sh all
```

为避免将数据或环境问题误判为模型能力差异，对数据集与 Gym 做了以下调整：

| 调整方向 | 主要内容 |
|---|---|
| 数据集整理 | 对齐任务描述、初始状态与工具配置；排除两个目标定义冲突、一个关键身份约束不可观察的任务。 |
| 判分一致性 | 修正状态检查的字段、比较与重复检查问题，使判分符合任务意图与数据库状态。 |
| 工具接口与错误处理 | 对齐工具声明与实际接口，保留错误语义，区分业务约束与基础设施故障。 |
| 状态与运行隔离 | 修复查询、写入和事务行为，保证返回证据与存储状态一致，隔离各条轨迹并及时释放资源。 |

详见 Gym 适配（[Backend compatibility](backends/enterpriseops/README.zh-CN.md)）。

### 故障类型

故障作用于工具调用边界，任务成功检查不变。五类机制下分七个 runtime 小类；数量来自 130 项 Base 固定 Test 清单（[base_test_fixed_mechanisms.json](artifacts/base_test_fixed_mechanisms.json)）。

| 机制 | 小类 / runtime 标签 | 注入表现 | Test 案例数 |
|---|---|---|---:|
| 1. 未执行写 | 暂时拒绝（`transient_unexecuted`） | 执行前拒绝，未产生写入。 | 26 |
| 2. 信息缺失 | 集合缺项（`partial_read`） | 移除部分条目，原始条目数（`count`）仍暴露缺失。 | 6 |
| | 响应截断（`truncated_read`） | 结构化读取中途截断。 | 20 |
| 3. 回包错配 | 响应串包（`reply_mixup`） | 将同一服务此前认可的响应附到另一读取。 | 28 |
| 4. 不确定提交 | 写入状态未知（`unknown_commit`） | 写入成功，确认替换为超时。 | 10 |
| | 创建状态未知（`unknown_commit_create`） | 创建成功，隐藏确认与生成 ID。 | 16 |
| 5. 筛选条件失效 — 分布外（OOD） | 可选谓词被忽略（`ignored_filter`） | 移除可选筛选，返回经验证的严格超集，保留必需查询范围。 | 24 |

对应恢复决策包括：对明确未执行的调用进行安全重试；对缺项或截断响应，通过补查完成读取恢复与证据补全；对串包核对响应与对象身份；在不确定提交后回读并核验写入状态，避免重复写入和其他重复副作用。这些决策用于工具调用失败恢复与有状态工作流恢复，保留已完成进展并继续剩余任务。

第五类整个机制族不参与本项目 SFT/RL，前四类参与训练。

这些机制对应真实服务中的失效模式：Microsoft Graph 会因限流[拒绝请求](https://learn.microsoft.com/en-us/graph/throttling)；Google Drive 明确标记[搜索结果可能不完整](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/list)，HTTP 连接中断也可能造成[响应未传完](https://www.rfc-editor.org/rfc/rfc9112.html#section-8)；GitHub 曾因连接复用错误发生[请求与数据库响应错配、私有数据泄露](https://github.blog/news-insights/the-library/incident-report-inadvertent-private-repository-disclosure/)；日历创建请求超时后重试可能重复创建，Microsoft Graph 因而提供[去重标识](https://learn.microsoft.com/en-us/graph/api/resources/event?view=graph-rest-1.0)；ServiceNow 默认会[忽略无效查询条件](https://www.servicenow.com/docs/r/api-reference/rest-apis/c_TableAPI.html)，可能扩大返回范围。由此可见，智能体若误判执行结果、证据完整性或操作范围，就可能遗漏任务、重复写入或修改错误对象。本项目将这些风险抽象为可复现的工具层注入机制，具体返回形式与发生频率不等同于生产环境。

### 可直接使用的教师产物

以下三类产物均由 DeepSeek-V4.1-Flash 辅助生成，默认配置直接读取仓库中的文件：

| 产物 | 提供的数据 | 用途 |
|---|---|---|
| 故障选择器结果 | [data/fault_selector/](data/fault_selector/) | Train/Val/Test 各任务与工具的候选故障标签，用于缩小运行时注入器的检查范围（[Fault injection](#故障注入流程)）。 |
| 检验依赖图 | [data/verifier_graph/](data/verifier_graph/) | Train/Val 的检验依赖，用于 RL 进度反馈（[Reward](#奖励reward以完成任务为目标为有效进展提供反馈)）。 |
| SFT 示范 | [data/sft/](data/sft/) | 692 条 Train 示范，以及用于快速监控 Val 集表现的 15 条小数据，提供 JSONL/Parquet（[SFT](#sft-1)）。 |

如需自行生成或采集，见故障选择器扫描（[Fault selector](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成故障标签)）、检验依赖图生成（[Verifier graph](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成检验依赖图)）和 SFT 示范采集（[Teacher collection](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)），其中给出 CLI、输出目录和后续使用方法。

## 4. 运行评估与准备教师数据

### 统一执行方式

Base 更倾向逐次调用工具，教师模型则更常在同一轮提出多条调用。为使教师示范和评估沿用 Base 的逐步交互方式，各模型均保留单次工具调用（`plan_preserving`）：每轮执行第一条工具调用，再根据真实返回重新推理。

上下文也按学生侧对齐：Base/SFT/RL 与教师评估统一用 Base tokenizer 和 chat template 计数，原任务 prompt、工具定义及完整 assistant/tool 历史都计入 32,768 token。每次请求前预留 2,048 输出 token，放不下就结束，不裁掉历史继续运行。

教师数据采集按去掉额外指导后的学生可见上下文计数，保留实际执行历史，再验收整条样本是否在 32K 内。全量采集每轮预留 4K 输出，评估与固定案例为 2K；教师额外指导不会占用导出学生样本的上下文。

### 故障注入流程

1. LLM 故障选择器缩小注入器的检查范围。选择器（[fault_candidates.py](src/robustops/data/fault_candidates.py)）根据任务、工具 schema 和原生返回形状，离线筛选可能适用的工具×故障组合。运行 clean 轨迹时，注入器只检查这些候选组合，避免对每个工具遍历全部故障机制；实际是否满足注入条件仍由真实调用、返回和状态决定。已有扫描结果（[fault_selector](data/fault_selector/)） · 生成扫描结果（[Fault selector](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行生成故障标签)）
2. 调度器选择机制并抽取重复次数上限。调度器（[fault_schedule.py](src/robustops/evaluation/fault_schedule.py)）先在实际合格机会中均衡选择一种机制，再按统一配置抽取其上限。Base 的本次分配可固定给后续模型使用。
3. 注入器核验并施加故障。检查参数、返回和状态，满足条件才注入并扣减剩余次数。例如“提交状态未知”必须先确认真实写入成功；选择器标签不保证实际调用合格。

故障选择器的输入包括原始任务与策略、工具的 schema 和服务器路由、verifier 元数据，以及从正常工具响应中提取的原生返回形状。它据此判断哪些故障可能适用于任务中的合法操作。

选择器提示词（[fault_selector.md](prompts/fault_selector.md)）采用以下原则：

- 依据任务原意、工具和原生返回形状判断，不虚构接口、字段或恢复能力。
- 允许合理替代路径；缺少尚未获取的 ID 或未来调用历史，不直接作为排除理由。
- 各故障独立判断，理由与标签一致，不为凑覆盖率选标签；实际能否注入由运行时核验。

详细原则见故障选择（[Fault selection](https://www.github.com/haohuiliang/RobustOps/blob/main/prompts/README.zh-CN.md#user-content-故障选择)）。

[data/fault_selector/](data/fault_selector/) 提供可直接使用的 Train、Val、Test 候选标签，由 DeepSeek-V4.1-Flash 根据上述输入离线辅助生成。标签供调度器和注入器使用，不作为额外提示加入执行模型上下文。

### 故障注入策略

没有固定故障注入配置时，先运行 clean 轨迹，由注入器检查调用是否满足注入条件，并保存合格调用前的状态与对话；再从这些机会中选择故障机制和注入位置，恢复保存点并回放所选调用，施加故障后继续执行。在选择器标签限定的候选范围内，优先覆盖实际注入较少的故障类型，以兼顾不同业务域与工具配置。

为不同机制设置重复注入次数上限，让模型学习不同故障强度下的应对方式。具体配置见故障注入策略（[Fault policy](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-故障注入策略)）。

### 使用现成配置评估

在固定 Val/Test 上配对评估无注入与故障条件，各模型共用对应分组的固定故障配置：[Val](artifacts/base_val_fixed_mechanisms.json) · [Test](artifacts/base_test_fixed_mechanisms.json)。如需在运行 Base 的 Val/Test 评估时，由本次调度生成并固定故障配置，见生成及后续评估的 CLI（[CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-重新生成并使用故障配置)）。

本地 BF16 评估建议最低配置为**单卡 16 GB（并发 1）**；默认**单卡 32 GB（并发 2）**。教师通过 API 评估，与 Base/SFT/RL 使用相同的任务、初始状态、工具、状态检查及上述执行预算，不追加教师指导。运行教师前按教师 API 配置（[Teacher API](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.zh-CN.md#user-content-教师-api-配置)）设置密钥。

```bash
# Test：分别评估 Base、SFT、RL 和教师。
bash run.sh inference --config recipes/inference/base_test.json
bash run.sh inference --config recipes/inference/sft_test.json
bash run.sh inference --config recipes/inference/rl_test.json
bash run.sh inference --config recipes/inference/teacher_test.json
# Val：分别评估 Base、SFT、RL 和教师。
bash run.sh inference --config recipes/inference/base_val.json
bash run.sh inference --config recipes/inference/sft_val.json
bash run.sh inference --config recipes/inference/rl_val.json
bash run.sh inference --config recipes/inference/teacher_val.json
```

本地模型的 GPU 数量可在配置中的 `hardware.gpus` 设置，也可用 `--gpus` 覆盖；`CUDA_VISIBLE_DEVICES` 选择具体设备。各配置的 `base_fixed_mechanisms` 可改为自己生成的 Base Val/Test 固定故障配置路径，参与对比的模型使用同一份配置。

### 教师数据采集

提供的教师数据采自 80% 可运行 Train 任务池，范围按任务 ID 确定。先以温度 0 运行，失败案例再以温度 0.6 补采一次。额外指导不占学生侧 32K 限长，去掉指导后验收完整样本。

教师先采集无注入轨迹，再根据实际可注入的机会均衡选择故障，从保存点继续执行；不采集 `ignored_filter`。具体流程与失败案例补采见采集指南（[Collection guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)）。

由统一的轨迹转 SFT 处理模块（[teacher_sft.py](src/robustops/data/teacher_sft.py)）读取带指导的教师模型推理产物，去掉教师指导，通过任务成功与上下文验收后，导出完整对话、监督 mask 和训练数据。系统消息、用户消息与工具返回不计入 loss；助手输出中，已识别的无效调用、不实叙述及仅供上下文的规划也不计入监督 loss。有效工具决策和故障后的恢复动作继续参与学习，错误与工具回执仍保留在上下文中。

具体屏蔽对象、保留行为与标注方式见 SFT loss mask 说明（[SFT loss masks](https://www.github.com/haohuiliang/RobustOps/blob/main/data/sft/README.zh-CN.md#user-content-教师导出的-loss-mask-与目标行为)）。

| 教师采集条件 | 有效 SFT 示范 | 对完整 Train 的示范覆盖率 |
|---|---:|---:|
| 无注入 | 357 条，覆盖 357 个组合 | **357/1,025（34.83%）** |
| 故障 | 335 条，覆盖 335 个组合 | **335/1,025（32.68%）** |

两者合计 692 条示范；289 个组合同时具有无注入/故障示范，去重后覆盖 **403/1,025 个组合（39.32%）**。

SFT 为 RL 提供冷启动。SFT 把教师成功解决任务的方法与执行路径加入学生的 policy pool，提高初始策略生成这些成功行为的可能性。它只使用采集池中教师在给定工具、上下文和故障约束下完成并通过验收的那部分样本；池内教师仍未解决的条件，以及池外未参与采集的任务，没有对应的 SFT 示范，留给 RL 在学生自己的推理与交互路径中探索。

#### 为什么增加教师指导？

早期 SFT 尝试使用未加教师指导的轨迹，再通过筛选、改写不实叙述和屏蔽错误动作改进监督。复核仍发现漏判和误删，改写文字也补不出未实际执行的正确路径；学生出现重复调用、无效循环和错误结束等行为。这些尝试的整体任务成功率反而低于 Base。

任务成功不代表整条轨迹都适合学习：教师可能没有查证就断言结果、叙述与实际调用不一致，或把部分完成说成全部完成。因此，指导不仅要求执行过程和叙述有依据，也约束可能被学生学到的坏习惯：说明保持简短，不编造操作；按请求的对象与范围行动，不因猜测扩大修改范围；有可靠结果后继续剩余任务，避免无必要的重复核验或在仍可执行时提前停下。

教师指导保留合理推断与探索，避免过多限制妨碍任务完成。SFT 教师指导提示词（[teacher.md](prompts/teacher.md)） 从基础任务和故障恢复两方面约束示范质量，再由状态检查和选择性监督决定哪些内容进入训练：

- 基础任务约束（无故障）：按原任务的目标、参数、范围和依赖完成工作，允许合理推断与探索；结果和完成声明要有可见依据，不编造操作或把部分完成说成全部完成，也不额外提高原任务要求。
- 故障恢复约束：根据可见证据选择恢复方式，例如明确未执行的暂时错误可有限重试、写入状态未知时先确认、读取不完整时补查。保留已完成的进展，继续剩余工作；可靠回执无需反复核验，不规定固定工具顺序或统一重试次数。

两类约束都不提供逐题规则、标准答案、状态检查内容或隐藏故障信息。

教师指导的编写原则是：

- 使用简短、通用、可观察的行为约束，减少学生可能模仿的错误。
- 依据任务原意和当前可见证据，允许合理推断与探索，不补造执行事实或提高完成要求。
- 保留有效尝试、恢复和结束行为，不规定僵硬的操作顺序。

详细原则见 SFT 引导（[SFT guidance](https://www.github.com/haohuiliang/RobustOps/blob/main/prompts/README.zh-CN.md#user-content-sft-引导)）。教师指导只用于辅助采集，导出学生输入和运行教师评估时均移除。

[data/sft/](data/sft/) 提供可直接使用的教师示范与监督 mask。DeepSeek-V4.1-Flash 根据原始任务、策略和执行中的工具返回，在教师指导下采集轨迹，再经任务成功、上下文验收与监督处理后导出。额外教师指导在导出时移除，学生输入保留原始任务与真实执行历史。

自行采集见数据指南（[Data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-自行采集-sft-示范)）。

<a id="training"></a>

## 5. 训练

先用教师示范训练 SFT，再从 SFT 检查点启动 RL。训练建议最低配置为**单卡 80 GB 显存**。

### SFT

```bash
# 使用全部可见 GPU 训练 SFT。
bash run.sh train --config recipes/training/sft.json
```

从 Arctic-MidTool-RL-4B 出发，使用 692 条成功教师轨迹。最终成功的轨迹仍可能经历无效调用，因此保留完整执行历史作为上下文，屏蔽已识别的无效决策与只作上下文的规划，让有效恢复继续参与学习。每条轨迹的监督权重在有效助手决策之间均分，再在各决策参与监督的 token 之间均分，避免较长的输出主导训练。

全参数 SFT 训练 3 轮，每步 8 条轨迹，学习率 3e-6。比较用于快速监控 Val 表现的小数据上的 loss，第 2 轮附近最低，继续到第 3 轮没有改善，因此采用 **第 2 轮 / 第 174 步** 的模型作为 RL 起点。自行准备监控数据见数据指南（[Data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-val-快速监控数据)）。

采用的 SFT 检查点在 Val 上的表现如下。Pass@1 是一次运行；Pass@4 是四次采样中至少一次完成任务：

| SFT Val 条件 | Pass@1（温度 0） | Pass@4（温度 0.6） |
|---|---:|---:|
| 无注入 | 24/127（18.90%） | 38/127（29.92%） |
| 实际注入故障 | 19/116（16.38%） | 30/123（24.39%） |

Pass@1/4 的差距说明单次执行仍有提升空间，也是继续 RL 的动机之一。Val 轨迹只用于评估，不参与策略更新。验证集轨迹（[Collections](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories#collections)） · Pass@4 采集说明与命令（[SFT Val Pass@4](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.zh-CN.md#user-content-采集-sft-val-pass4-轨迹)）

### RL

```bash
# 使用全部可见 GPU，从 SFT 训练 RL 至 step 120。
bash run.sh train --config recipes/training/rl.json
```

GPU 配置、检查点保存、续训和重训权重评估见训练指南（[Training guide](recipes/training/README.zh-CN.md)）。

从采用的 SFT 出发训练 120 步，每步 32 组 × 4 条：16 组无注入、16 组故障。温度 1.0、学习率 1e-6，相对 SFT 的 KL 系数 0.001。两种条件同时训练，因为故障恢复仍依赖基础业务任务的完成能力。

#### 从同一个故障后状态比较恢复决策

无注入组从任务初态开始。故障组先由当前策略采集一条 clean 轨迹，记录真实合格调用前的业务状态与对话，再从可注入机会中选择故障机制和重复次数上限。恢复所选保存点、执行调用并实际注入故障后，同组四条轨迹从相同的故障后状态继续，比较它们如何解决同一个恢复问题。

```text
当前策略采集 clean → 记录合格注入点 → 选择机制与重复次数上限
                                           ↓
                         恢复保存点并注入 → 相同故障后状态
                                           ↓
                                采样 4 条续行 → 检查与更新
```

只对新生成的决策计算策略损失，继承的前缀提供上下文。这个设计聚焦故障后的恢复决策；故障边界之前的预防行为，不在该故障组的直接训练范围内。

探索首先要产生可执行的替代路径。早期尝试将温度升到 1.5 后，出现协议错误和长段混杂输出；相同首批 48 条无注入轨迹在温度 0.6 / 1.5 / 1.0 下分别成功 11 / 0 / 11 条。因此回调到 1.0：增加随机性不一定能带来有效探索。

SFT 学习固定教师历史下的示范动作，RL 则在当前策略自主到达的状态上，用组内相对反馈更新决策。同一任务可以产生新的错误、故障位置和恢复路径，所以两阶段允许任务重叠：既改进已有示范行为，也探索教师没有覆盖的状态。

#### 奖励（`reward`）：以完成任务为目标，为有效进展提供反馈

仅用完整任务成功作为奖励时，信号过于稀疏：例如，早期一次 10 步实验的 320 组采样中，223 组（69.69%）四条轨迹全部失败，单靠成败无法区分组内优劣。为增加中途反馈，尝试过使用 verifier 统计部分进展，但独立累计通过项又可能奖励错误父对象或业务上下文下的操作。最终方案采用检验依赖图（[Verifier graph](#为什么需要检验依赖图)）门控后的净进展：奖励前提成立的新增成果、扣除丢失成果，并排除续行起点已有进展。

$$
\begin{aligned}
R &= S + 0.3P + 0.1Q, \\
S &= \mathbf{1}[\text{all verifier checks pass}], \\
P &= \frac{N_{\mathrm{passed}}^{\mathrm{dep}}(\mathrm{final})-N_{\mathrm{passed}}^{\mathrm{dep}}(\mathrm{start})}{N_{\mathrm{verifier}}}, \\
Q &= S\left(1-\frac{B}{\max(1,C)}\right).
\end{aligned}
$$

`N_verifier` 为 verifier 检查项总数；`N_passed^dep` 只统计自身及所有必要前提均通过的检查项。start 为本次续行起点，final 为结束状态。

`S`、`P`、`Q` 是未经组内比较的原始奖励分项：`S` 衡量完整业务成功；`P` 奖励依赖关系成立的新进展、扣除状态退化；`Q` 衡量成功轨迹中的调用质量，`B` 为经证据确认的坏调用数，`C` 为后缀工具尝试数。

统一的 [GRPO](https://arxiv.org/abs/2402.03300) 训练路径通过 `training.reward_normalization` 选择两种归一化顺序：`aggregate` 先合成原始总奖励，再组内标准化；`component` 先分别标准化各奖励分项，再加权，采用 [GDPO](https://arxiv.org/abs/2601.05242) 的分项归一化方式。两种方式共用按轨迹等权的批次白化和局部信用实现：

$$
\begin{aligned}
A_i^{\mathrm{aggregate}} &= Z_{\mathrm{batch}}\left[Z_{\mathrm{group}}(R_i)\right], \\
A_i^{\mathrm{component}} &= Z_{\mathrm{batch}}\left[Z_{\mathrm{group}}(S_i)+0.3Z_{\mathrm{group}}(P_i)+0.1Z_{\mathrm{group}}(Q_i)\right], \\
Z(x_i) &= \frac{x_i-\mathrm{mean}(x)}{\mathrm{std}(x)+\epsilon}.
\end{aligned}
$$

`Z_group` 在同组四条轨迹间计算；`Z_batch` 对接受批次的轨迹优势减去均值、除以样本标准差。默认使用 `component`，成功、进展、质量的权重为 `[1.0, 0.3, 0.1]`。在另一份配置中设为 `aggregate` 即可比较两种归一化顺序。原始奖励相同的轨迹，放入不同组可能得到不同优势；失败轨迹也可能因组内相对进展较好而获得正优势。配置和续训方式见[训练指南](recipes/training/README.zh-CN.md)。

损失平均借鉴 DAPO 的 token 级聚合（[DAPO](https://arxiv.org/abs/2503.14476)）：分别将无注入与故障两类的有效生成 token 损失求和、除以各类 token 总数，再让两类各占一半权重。工具返回、继承前缀和 padding 不计入分母；完整目标与实现见训练指南（[Training guide](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/training/README.zh-CN.md#user-content-优势与策略损失的实现路径)）。

#### 为什么需要检验依赖图？

独立累计通过的检查，可能在必要前提缺失时仍奖励下游结果。关键在于“做成的这一项，是否建立在正确前提上”：检验依赖图表达任务检查之间的必要业务关系，让部分进展的信用与这些前提一致。

例如，Calendar 任务检查 A：创建次级日历、B：授予 Carol 编辑权限、C：安排启动会议。图中设定 `A → B` 和 `A → C`：若 B、C 通过而 A 失败，独立计数是 `2/3`，门控后为 `0/3`；若 A、B 通过，则为 `2/3`。训练中的 `P` 计算这种有效进展的增减，完整成功仍由检查判定。

检验依赖图生成方案也经历过比较：早期尝试把复合检查拆成更细的 SQL 谓词，能提供更细的反馈，但容易加入原成功标准之外的要求。最终方案保留任务检查，只补必要的前置依赖；保留原成功标准，同时接受部分进展仍受检查粒度限制的取舍。

检验依赖图生成的输入包括原始任务与策略、带索引的 SQL 检查及其比较符和期望值，以及相关数据库 schema；不提供初始数据行或解题轨迹。

检验依赖图提示词（[verifier_graph.md](prompts/verifier_graph.md)）采用以下原则：

- 按 SQL、比较符和期望值判断，不新增检查或提高成功标准。
- 只连接必要的业务前提，保留独立操作的进展。
- 每条依赖给出证据，不确定时省略，允许空图。

程序检查无环，并验证图满分与检查全部通过等价。详细原则见检验依赖图生成（[Verifier graph](https://www.github.com/haohuiliang/RobustOps/blob/main/prompts/README.zh-CN.md#user-content-检验依赖图生成)）。

[data/verifier_graph/](data/verifier_graph/) 提供可直接使用的图，由 DeepSeek-V4.1-Flash 根据任务、检查和 schema 离线辅助生成。SQL、期望值和图元数据不进入执行模型的上下文。

#### 局部信用：较好的轨迹也有不该鼓励的动作

轨迹间的比较仍不能代替动作间的判断。最终方案对已确认的坏调用，以及没有新信息或状态变化的重复读取片段施加负信用；重复读取从第三次起标记，保留合法重试与状态核验。失败案例也可能因为相对较好的进展获得正优势，但不能因此同样鼓励“任务未完成就结束”：仅将主动结束片段的优势上限设为零，保留此前有效推进的信用。

$$
A'_{i,t} = \min(A_{i,t},-\eta), \qquad \eta=0.5, \quad t\in\mathcal{B}_i,
$$

$$
A'_{i,t} = \min(A_{i,t},0), \qquad t\in\mathcal{T}_i.
$$

$`A_{i,t}`$ 是局部覆盖前的 token 优势，已包含质量信用的时间分配；$`\mathcal{B}_i`$ 是已标记坏调用与无进展重复读取的 token 集合，$`\mathcal{T}_i`$ 是失败轨迹主动结束片段的 token 集合。第一式保证坏动作获得不高于 −0.5 的优势，第二式去掉失败结束的正优势，已有负优势保持；其余位置不作这两项覆盖。

奖励实现（[rl_credit.py](src/robustops/training/rl_credit.py)） · 归一化（[rl_advantage.py](src/robustops/training/rl_advantage.py)） · 局部信用（[recovery_local_credit.py](src/robustops/training/recovery_local_credit.py)）

具体处罚对象、触发证据与保留行为见局部信用说明（[Local credit](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/training/README.zh-CN.md#user-content-局部信用的目标行为)）。

#### 避免过早结束：让轨迹继续完成任务

延长轨迹的目的是让智能体在仍有可执行步骤时继续完成任务。历史复核中，已有工具暴露且执行预算仍有余量，模型却声称无法继续；也有只完成部分要求便结束的案例。减少这类提前停止，才有机会补查信息、处理故障并完成剩余工作。

早期尝试对失败的主动结束施加强负信用，希望促使模型继续，但更长执行中也出现反复猜参数、重复读取和耗尽预算。最终采用上文的停止优势上限，去掉失败结束的正信用，保留原有负优势与此前有效推进的信用。成功完成任务的轨迹则提供完成后结束的学习信号。完成更多工作、成功后及时停止，都是执行成本需要同时考虑的目标。

#### 样本筛选：有相对差异才有比较价值

早期加入过长度差异奖励，希望成功案例在完成任务的同时找到更短的路径。但它也使 72/480 个接受组（15%）仅凭长度差异进入训练；另一种早期入选方式则在成功、进展和质量奖励均无组内差异时，只因局部处罚产生非零策略损失便保留整组。两者都占用有限的训练批次，减少了用于比较任务完成与有效进展的组。最终移除长度项，根据奖励在组内能否形成非零相对优势筛选，排除全成功组，再对入选轨迹施加局部信用。

DAPO 的动态采样（[DAPO](https://arxiv.org/html/2503.14476v1#S3.SS2)）过滤缺少组内相对信号的样本并继续采样，补足训练批次。本项目根据奖励判断相对信号，全失败组只要进展不同，仍然可以入选。

在动态补采之外，本项目还设计了**零信号案例的延迟回流机制**：暂时没有比较价值的案例先离开当前批次，等待策略更新后再尝试。零信号组在第 k 步被移出后，最早于第 k+10 步重新参与补位；只有后续批次需要替换被筛除的组时，才从已到期队列中取出相同无注入或故障类型的案例，并避开本批已使用的案例。无注入案例从头执行，故障案例从保存的恢复边界继续，都由当前策略重新采样轨迹，而非重用旧轨迹进行更新。若仍无信号，则从本次步骤起再等待十步；全成功组不进入该回流队列。

#### 直接相关的训练工作

| 工作 | 与本项目实现的关系 |
|---|---|
| [GDPO — Liu et al., 2026](https://arxiv.org/abs/2601.05242) · 官方代码（[NVlabs/GDPO](https://github.com/NVlabs/GDPO)） | 每个奖励分项先组内归一化，再加权及批次归一化；本项目使用成功、进展和质量奖励，按轨迹等权白化优势；loss 另按 token 聚合。 |
| [DeepSeekMath / GRPO — Shao et al., 2024](https://arxiv.org/abs/2402.03300) | 相同输入多条续行的组内相对优势，不训练价值评估模型（`critic`）；本项目 G=4，故障同组共享恢复边界。 |
| [DAPO — Yu et al., 2025](https://arxiv.org/abs/2503.14476) | 借鉴 token 级 loss 聚合和动态采样；本项目另有 clean/fault 权重平衡与延迟回流，保留对称裁剪和 KL，未复刻完整 DAPO。 |
| [HybridFlow — Sheng et al.](https://arxiv.org/abs/2409.19256) · [veRL](https://github.com/verl-project/verl) | 分布式训练与轨迹采样基础设施，通过固定版本项目补丁扩展。 |

#### 首采奖励曲线

![首采奖励](assets/figures/initial_reward.png)

过滤缺少组内相对信号的组之前的平均奖励。浅色线为逐步均值，蓝线为后向 10 步均值；前 10 步与最后 10 步的均值分别为 0.457 / 0.522。[SVG](assets/figures/initial_reward.svg)

使用以下命令查看 120 步训练的 24 项 TensorBoard 指标：

```bash
tensorboard --logdir logs
```

<a id="evaluation"></a>

## 6. 评估

以下汇总固定 Test 上的任务成功、故障覆盖与 OOD 表现。评估使用任务与 verifier，并在公共实际命中集合上比较故障条件下的任务成功。

### Base / SFT / RL 评估结果

对未实际触发故障的案例补采，提高故障覆盖率；下表同时报告覆盖率与任务成功，并在公共实际命中集合上比较各模型。

| 模型 | 无注入任务成功 /130 | 最终注入命中 /130 | 最终实际命中后的任务成功 | 公共命中任务成功 /118 |
|---|---:|---:|---:|---:|
| Base | 15/130（11.54%） | 128/130（98.46%） | 9/128（7.03%） | 9/118（7.63%） |
| SFT | 28/130（21.54%） | 121/130（93.08%） | 29/121（23.97%） | 28/118（23.73%） |
| **RL120** | **40/130（30.77%）** | 125/130（96.15%） | 42/125（33.60%） | **40/118（33.90%）** |

RL120 在相同的 125 项中，有 9 项仅故障条件成功、7 项仅无注入成功，说明两种条件下的成功路径并不完全重合。轨迹复核显示，故障反馈可以促进补查信息与调整执行路径，让模型探索无注入执行未发现的成功方案。

### 对未见故障机制的泛化（OOD）

筛选条件失效（`ignored_filter`）未参与本项目的 SFT/RL 训练，用于检验模型能否将学到的恢复策略迁移到未见故障机制。

| 模型 | 相同 OOD 公共命中集合上的任务成功 /21 |
|---|---:|
| Base | 0/21（0.00%） |
| SFT | 4/21（19.05%） |
| **RL120** | **7/21（33.33%）** |

在这 21 项上，RL 保留了 SFT 原有的全部 4 项成功，并新增 3 项，表明学到的核验与任务推进方式在这一未见故障机制上具有泛化能力。

原始配对评估不含补采，RL120 比 SFT 多使用 30.63% 生成 token 和 30.86% 工具调用。减少冗余补查、改善停止时机，是兼顾完整交付与执行效率的潜在方向。

### 轨迹证据

[RobustOps-Trajectories](https://huggingface.co/datasets/haohuiliang/RobustOps-Trajectories) 包含 Base、带指导的教师采集、无额外指导教师评估、SFT 和 RL 的执行轨迹，提供工具调用与返回、故障事件、状态检查查询值，以及 SFT 数据对应的教师来源，支持从最终结果回看具体决策过程。

## 模块与代码入口

项目由七个模块组成，通过 [run.sh](run.sh) 和 [recipes](recipes/) 配置启动：

| 模块 | 代码 | 职责 |
|---|---|---|
| 业务环境 | 业务环境（[enterpriseops](src/robustops/envs/enterpriseops/)） · 后端适配（[enterpriseops](backends/enterpriseops/)） | 提供 MCP 工具、隔离的任务状态与成功检查。 |
| 故障流程 | 选择器（[fault_candidates.py](src/robustops/data/fault_candidates.py)） · 调度器（[fault_schedule.py](src/robustops/evaluation/fault_schedule.py)） · 注入器（[selected_fault.py](src/robustops/evaluation/selected_fault.py)） | 初筛可能适用的机制、分配有限预算，并在实际调用中检查条件、记录命中。 |
| 检验依赖图（`verifier graph`） | 依赖图生成（[build_verifier_graphs.py](scripts/build_verifier_graphs.py)） · 进度门控（[recovery_verifier_graph.py](src/robustops/training/recovery_verifier_graph.py)） | 建立检查间的依赖，计算有效的部分进展。 |
| 数据准备 | 数据处理（[data](src/robustops/data/)） · 任务划分（[task_sets](data/task_sets/)） | 准备任务，采集和验收轨迹，导出监督掩码。 |
| 监督微调（SFT） | 决策训练（[decision_sft_trainer.py](src/robustops/training/decision_sft_trainer.py)） | 保留执行上下文，学习有效决策。 |
| 强化学习（RL） | 归一化管理（[rl_manager.py](src/robustops/training/rl_manager.py)） | 采样当前策略续行，利用状态检查反馈与局部信用更新。 |
| 推理与评估 | 评估执行（[evaluation](src/robustops/evaluation/)） | 运行本地/API 案例和固定 Val/Test，统计任务成功、注入覆盖、恢复与 OOD。 |

## 仓库内容与许可

`src/` 为实现；配置（[Recipes](recipes/README.zh-CN.md)）、数据（[Data](data/README.zh-CN.md)） 与 提示词（[Prompts](prompts/README.zh-CN.md)） 说明配置、数据和案例推理。评估汇总（[Evaluation](#6-评估)）见本文；`logs/` 与 `assets/figures/` 提供训练曲线。

项目代码使用 [Apache-2.0](LICENSE)，基于 [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)、[veRL](https://github.com/verl-project/verl) 与 [Arctic-MidTool-RL-4B](https://huggingface.co/MidTool/Arctic-MidTool-RL-4B)，保留上游许可与归属（[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。贡献在恢复环境、训练集成、评估工程与实验分析。
