# 五个 Base / RobustOps 对比示例

[English](README.md) · [简体中文](README.zh-CN.md)

无需运行完整评测，直接对比两个 clean 任务和三个 fault 任务。每例固定任务、初始数据库、工具和状态校验；两个模型分别从重置后的任务状态开始。这组示例用于观察执行过程，整体能力请参考[完整评测](../README.zh-CN.md)。

## 准备与运行

先按首页完成环境和企业 MCP 运行时配置，将 Base 下载到 `models/base/Arctic-MidTool-RL-4B`，将 RL BF16 下载到 `models/rl`。不需要 SFT 模型或教师 API 密钥。在仓库根目录、激活 `rops` 环境后执行：

```bash
# 只检查任务配置，不启动模型和 MCP。
bash scripts/demo.sh --dry-run
# 单张 GPU 依次运行 Base、RL 的全部五例。
bash scripts/demo.sh
# 只运行一个案例；新选择使用独立输出目录。
bash scripts/demo.sh --case calendar_retry --output runs/demo_calendar
# 也可分别运行两个模型。
bash scripts/demo.sh --model base --output runs/demo_separate
bash scripts/demo.sh --model rl --output runs/demo_separate
# 从已有结果重新生成报告，无需 GPU 或 MCP。
bash scripts/demo.sh --report-only
```

默认一次运行一条轨迹，建议使用 32 GB 显存的 GPU，以容纳 32K 上下文。可用 `CUDA_VISIBLE_DEVICES` 选择显卡。运行器自动启动需要的服务，结束后停止自己启动的模型和 MCP 进程。

重复传入 `--case` 可选择多个案例。更换配置或案例组合时指定新的 `--output`。相同配置会复用已保存的有效轨迹，不会为了获得成功而重试失败任务。遇到无效运行时保留原记录，修复原因后换目录运行。

## 五个案例

| ID | 条件 | 任务 | 故障 |
|---|---|---|---|
| `teams_channels` | Clean | 检查并创建工程频道，归档旧合作频道 | 无 |
| `drive_access` | Clean | 批准文档访问申请，配置协作权限和元数据 | 无 |
| `itsm_reply` | Fault | 修正设备价格并创建紧急事件 | 一次符合条件的读取收到错配回复 |
| `calendar_retry` | Fault | 创建日历、配置属性并安排会议 | 最多四次符合条件的写入在执行前被拒绝 |
| `hr_filter` | Fault | 迁移 ready 状态的关键优先级 HR 工单并调整负责人 | 一次符合条件的读取忽略可选筛选条件 |

两边使用相同故障规则，但模型可能选择不同工具，因此注入位置可能不同。配置了故障却未实际命中的结果会单独标记。

任务引用的是预置数据库中的对象，不能只改提示词就沿用原来的成功校验。自定义任务时需要同时调整任务数据、工具和校验。完整任务要求保存在运行后的 `plan.json` 和报告里。

## 查看结果

运行结束后，脚本会打印 `runs/demo/comparison.txt` 的路径。可用文本编辑器打开，也可在终端查看：

```bash
less runs/demo/comparison.txt
```

文件开头是简洁的 Base / RL 结果总表，随后按案例展示未通过的校验、故障位置、最终回答摘要和终止原因。完整任务、最终回答和工具轨迹见对应的 `details/<case_id>.txt`。详情中的较长工具参数和返回值会缩略，原始 JSON 保留完整内容。

完整原始记录保留在 `<output>/{base,rl}/<case_id>/{clean,fault}/` 下的 `attempt.json` 和 `calls/` 中；`comparison.json` 提供结构化结果，模型目录还保留配置与运行日志。

`PASS` 要求运行有效、数据库清理成功、任务校验全部通过，fault 案例还必须实际命中故障。`FAIL`、`INVALID`、`NO FAULT HIT`、`NOT RUN` 分别表示任务失败、运行无效、故障未命中和未运行。状态校验通过并不等于正常结束或已经生成最终回答，因此报告另外显示终止原因和调用次数。

[案例清单](../data/cases/demo.json)、[Base 配置](../recipes/inference/demo_base.json)和 [RL 配置](../recipes/inference/demo_rl.json)定义模型路径和运行设置，并复用现有案例执行器。
