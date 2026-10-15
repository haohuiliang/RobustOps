# EnterpriseOps runtime adapters

[English](README.md) · [简体中文](README.zh-CN.md)

RobustOps uses the original [EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym/tree/271f2c357f763376997dfd16807fcde2474ae41b) simulator with adapters for calendar, email, drive, CSM, HR, ITSM and teams.

## Prepare the runtime

Run from the repository root after the [environment setup](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-1-install-the-environment).

```bash
# Prepare the seven-domain MCP runtime and project adapters.
bash backends/enterpriseops/prepare_udocker.sh all
```

The inference runner starts the required services and cleans up each isolated database. The default backend uses rootless udocker; a Docker launcher is also provided.

## Project backend improvements

Backend changes make the tools' advertised contracts agree with execution:

- **Tool contracts:** [Teams](bootstrap_teams.py) exposes executable catalog entries and nested message-body requirements; [Drive](bootstrap_drive.py) validates supported label operations and parent-only moves.
- **Failure evidence:** [shared error handling](backend_errors.py) preserves typed HTTP/MCP failures and maps request-owned database constraints to explicit rejections. Internal key failures, database locks and service failures remain infrastructure errors. Concurrent calls keep their own error metadata.
- **State integrity:** [Email](bootstrap_email.py) creates history parents before dependent writes and preserves the sent message when deleting its draft wrapper. Drive label changes validate the whole request before one commit. [Calendar](bootstrap_calendar.py) keeps ACL sessions alive through serialization, then closes them and releases pending transactions.
- **Read semantics:** Email applies all-label filtering; Calendar distinguishes recurring roots, instances and exceptions before pagination and preserves timezone meaning. Teams rejects unsupported filters. [HR](bootstrap_hr.py) avoids generated knowledge-number collisions without changing valid requests.

The guiding principles are to preserve real commits and API semantics, expose recoverable business rejections, isolate infrastructure failures, and release resources at the owning request boundary. Repairs use general schema/storage rules, without task IDs, reference answers or verifier-guided argument repair. Each attempt has an isolated database and the task-success verifier; controlled faults are applied separately at the tool boundary.

[Reproduction](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-3-prepare-the-dataset-and-enterprise-runtime) · [Upstream setup](https://github.com/ServiceNow/EnterpriseOps-Gym)
