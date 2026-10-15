# EnterpriseOps 运行环境适配

[English](README.md) · [简体中文](README.zh-CN.md)

RobustOps 使用原始 [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym/tree/271f2c357f763376997dfd16807fcde2474ae41b) 模拟器，适配 calendar、email、drive、CSM、HR、ITSM 和 teams 七域。

## 准备运行环境

先完成环境安装（[Setup](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-1-安装环境)），在仓库根目录执行：

```bash
# 准备七个业务域的 MCP 运行环境及项目适配。
bash backends/enterpriseops/prepare_udocker.sh all
```

推理运行器启动所需服务，并清理每条轨迹的独立数据库。默认使用无 root 权限的 udocker，也提供 Docker 启动器。

## 本项目的后端优化

后端优化使工具声明与实际执行一致：

- **工具契约**：[Teams](bootstrap_teams.py) 暴露可执行的工具目录与消息体嵌套要求；[Drive](bootstrap_drive.py) 校验支持的标签操作，并支持只修改父目录的移动请求。
- **错误证据**：[统一错误处理](backend_errors.py) 保留 HTTP/MCP 错误类型，将请求导致的数据库约束冲突转为明确拒绝；内部生成键错误、数据库锁和服务故障仍归为基础设施错误。并发调用的错误元数据独立保存。
- **状态完整性**：[Email](bootstrap_email.py) 先创建历史父记录再写入子记录，删除草稿包装时保留已发送邮件；Drive 标签变更先校验完整请求，再一次提交。[Calendar](bootstrap_calendar.py) 将 ACL 会话保留到响应序列化结束，再关闭并释放未提交事务。
- **读取语义**：Email 按全部标签进行筛选；Calendar 在分页前区分重复事件的主记录、实例和例外，并保留时区含义。Teams 拒绝不支持的筛选表达式；[HR](bootstrap_hr.py) 避免自动生成知识编号冲突，不改变合法请求。

原则是保留真实提交与 API 语义，让业务拒绝可恢复，将基础设施失败单独处理，并在拥有资源的请求边界释放资源。修复依赖通用 schema 与存储规则，不使用任务 ID、参考答案或 verifier 替模型修正参数。每条轨迹使用独立数据库和任务成功检查（`verifier`）；受控故障另在工具调用边界施加。

[数据与业务环境准备](https://www.github.com/haohuiliang/RobustOps/blob/main/README.zh-CN.md#user-content-3-准备数据集与业务环境) · [上游环境说明](https://github.com/ServiceNow/EnterpriseOps-Gym)
