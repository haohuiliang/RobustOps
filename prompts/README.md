# LLM prompts and fixed cases

[English](README.md) · [简体中文](README.zh-CN.md)

These three prompts support SFT teacher collection, fault selection and verifier-graph construction, all using **DeepSeek-V4.1-Flash**. Task system/user prompts stay in the task assets. Scripts read the files directly; runtime evaluation adds none of these instructions to the agent.

| Prompt | Inference input | Inference output | Consumer |
|---|---|---|---|
| [SFT teacher guidance prompt](teacher.md) | Original task and policy, tools, and the evolving assistant/tool history. | Executed tool calls and a final response; accepted trajectories become SFT records after guidance removal and loss masking. | `teacher_reproduction` and `case_inference` via `teacher_guidance()`. |
| [Fault selection prompt](fault_selector.md) | Task/policy, tool schemas, server routes, verifier metadata and observed native response shapes. | JSON covering every tool: role, reason, potentially applicable fault mechanisms, recovery tools and safe-replay basis. | `fault_candidates` / `fault_dataset`. |
| [Verifier graph prompt](verifier_graph.md) | Task/policy, indexed SQL checks with expected values/comparisons, and relevant database schema. No seed rows or solver trajectory. | JSON retaining every verifier node, with prerequisite indices, evidence and uncertainties. | `scripts/build_verifier_graphs.py`. |

## SFT guidance

Guided inference artifacts are processed by the unified [trajectory-to-SFT module](../src/robustops/data/teacher_sft.py): guidance removal, supervision masks and training-data export. See the [processing CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-convert-trajectories-to-sft-data).

Use the supplied [SFT data](../data/sft/README.md) directly, or follow the [collection CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-generate-or-collect-your-own) to collect your own.

Early postprocessing-only trials still missed errors, removed valid content and could not supply correct paths that had never been executed, motivating general guidance during collection. The guidance aims to produce suitable SFT demonstrations by reducing invented outcomes, contradictions between narration and actions, and omitted remaining work. It preserves task scope and dependencies while allowing reasonable inference, exploration and independent work, without task-specific rules, reference answers or hidden fault information. Recovery is conditional: retry explicitly unexecuted calls within bounds, inspect uncertain writes before risking duplicates, and cross-check missing or conflicting evidence. Reliable receipts do not require redundant checks.

More restrictive variants did not establish a stable supervision-quality gain, and unguided controls sometimes completed more tasks. The project retained general guidance and reviews executed decisions rather than accumulating task-specific exceptions. [Teacher diagnostics and decisions](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-why-add-teacher-guidance) explain that tradeoff. The adopted text is unchanged from teacher collection. `consistent_narration` loads it; `none` returns an empty string. Only `teacher_data` adds it to the original system prompt. Collection counts student-visible context after removing guidance. Export restores the original prompt and checks that the full sample fits within 32K; evaluation always omits guidance. See the [shared execution protocol](https://www.github.com/haohuiliang/RobustOps/blob/main/README.md#user-content-shared-execution-protocol) for context alignment.

### Prompt-writing principles

- Target observable behavior that is useful for supervision: concise explanations, agreement between actions and narration, and completion claims supported by visible evidence.
- Follow the requested objects, scope and dependencies while allowing reasonable inference and exploration. Do not invent execution or add completion requirements.
- Trigger recovery from evidence: retry calls known not to have executed, confirm uncertain writes, and complete incomplete reads. Preserve progress and continue remaining work.
- Avoid redundant checks after reliable receipts and premature stopping while executable work remains. Do not prescribe a fixed tool order or uniform retry count.
- Use general rules without case-specific answers, verifier contents or hidden fault information. Remove guidance from student data and teacher evaluation.

[data/sft/](../data/sft/) provides ready-to-use teacher demonstrations and supervision masks. DeepSeek-V4.1-Flash collects guided trajectories using the original task, policy and tool responses during execution. Task-success and context checks plus supervision processing determine the exported data. Extra teacher guidance is removed during export; student inputs retain the original task and actual execution history.

## Fault selection

The fault selector receives the original task and policy, tool schemas and server routes, verifier metadata, and native response shapes extracted from normal tool responses. These inputs support judgments about faults that could apply to legitimate task operations.

The selector preliminarily screens all seven labels for each tool to narrow the candidate range, with reasons and possible recovery routes. It must cite tools and supported semantics, including native count evidence for partial reads and a feasible read-back or safe replay for uncertain writes. It does not invent interfaces or select tools simply to reach a coverage target.

The runtime injector determines actual applicability from arguments, results, history and remaining injection allowance. The `ignored_filter` family is excluded from project SFT/RL training; Val/Test can use it to evaluate generalization to an unseen fault mechanism. Supplied response-shape observations describe earlier normal responses; they are not additional fault labels.

Selector scan results in [data/fault_selector/](../data/fault_selector/) include Train, Val and Test. Per-tool labels constrain later fault attempts; the scheduler selects a mechanism and budget, and the injector checks actual calls. Test tool labels are also saved in the [fixed evaluation record](../artifacts/test_fault_policy/policy.json), matching [test.jsonl](../data/fault_selector/test.jsonl); this is not the final mechanism assignment. Generate your own labels below, then follow the [data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-generate-fault-labels) for bindings:

```bash
# Regenerate candidate labels using DeepSeek-V4.1-Flash into runs/fault_selector/dataset.
bash run.sh inference --config recipes/inference/fault_selector.json
```

### Prompt-writing principles

- Ground legitimate task-relevant uses in exposed schemas, actual server routes and native response shapes. Do not invent fields, endpoints or capabilities, or infer semantics from tool names alone.
- Allow alternative plans and IDs obtainable through tools. Missing observations, future history or an empty current result do not automatically imply missing capability; uncertainty is not positive evidence either.
- Require an executable recovery path: name exposed readback tools and explain how their inputs remain obtainable after receipt loss, or provide a supported safe-replay basis. Parent metadata does not establish child effects.
- Check every fault independently for each tool, writing evidence before consistent labels. Allow empty labels without task-specific patches or coverage quotas.
- Labels only define candidates. The injector verifies actual arguments, returns, counts, history and state. Task verifiers aid task interpretation; hidden values are not agent knowledge.

[data/fault_selector/](../data/fault_selector/) provides ready-to-use Train, Val and Test candidate labels, generated offline with DeepSeek-V4.1-Flash from these inputs. The scheduler and injector consume the labels; they are not added to the execution model's context as extra instructions.

## Verifier dependency graph generation

Verifier dependency graph generation receives the original task and policy, indexed SQL checks with their comparisons and expected values, and the relevant database schema. It does not receive initial data rows or solution trajectories.

The graph addresses misleading partial progress: a downstream check should not earn credit when a prerequisite for its intended context is unsatisfied. It adds dependencies between checks, without new SQL, additional predicates or changed success labels. Independent edits, negative results and cleanup remain independent unless the measured contract requires a prerequisite. Uncertain edges are omitted; an empty graph is valid.

Earlier experiments explored finer SQL checks; the adopted prompt instead preserves the nodes and equal weights. It reads each Boolean condition literally, then asks whether another original check supplies missing task context, functional eligibility or an explicitly asserted completed effect. A shared table, operation order or generic completion message is not sufficient evidence for an edge.

The prompt balances two risks: omitting necessary context rewards work for the wrong subject, while over-linking suppresses valid independent progress. A single-object check may anchor a necessary parent or role even when it bundles other fields; those fields must be disclosed. It must not be used solely to demand full configuration, uniqueness or existence already established by the dependent query. Each edge needs evidence, and transitive dependencies must not import unrelated work. If the needed finer check does not exist, report that limitation instead of creating one.

### Prompt-writing principles

- Interpret the SQL, comparisons and expected values, preserving every node and its equal weight. Do not add SQL, checks or stronger success criteria.
- Connect only necessary context: a measured relation's parent, beneficiary or role eligibility, or the fact asserted by an explicit completion claim. Shared tables, operation order and generic completion messages are insufficient evidence.
- Preserve independent field edits, quantities, negative results and cleanup. Do not gate existence already established by the query. A coarse check can supply necessary context, but disclose its bundled independent fields.
- Justify every edge and inspect transitive dependencies for unrelated work. Omit uncertain edges, allow an empty graph and report missing bindings; do not claim the graph repairs ambiguous SQL joins.

The host validates node coverage, acyclicity and mode ordering, then checks full-success equivalence. With Boolean results `v_i`, gated results are `g_i = v_i AND all(g_j for prerequisites j)`. Their mean supplies progress credit; final task success still requires all checks. The graph cannot repair an ambiguous SQL join or supply a finer check absent from the task-success criteria.

The supplied [Train/Val verifier graphs](../data/verifier_graph/) are ready to use. To generate your own:

```bash
# Regenerate Train/Val graphs using DeepSeek-V4.1-Flash into runs/verifier_graph/dataset.
python scripts/build_verifier_graphs.py --config recipes/data/verifier_graph.json
```

Generation freezes the prompt and request/response records in its run directory. Resume graph requests with `--phase run`, then export with `--phase export`; a changed prompt requires a new run directory. Generated labels and graphs go to `runs/`, leaving supplied data intact. See the [data guide](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-generate-or-collect-your-own) for configuration bindings.

See [teacher API configuration](https://www.github.com/haohuiliang/RobustOps/blob/main/recipes/README.md#user-content-teacher-api-configuration) for API key setup.

[data/verifier_graph/](../data/verifier_graph/) provides ready-to-use graphs, generated offline with DeepSeek-V4.1-Flash from tasks, checks and schema. SQL, expected values and graph metadata do not enter the execution model's context.

## Run a case

Local models and the API teacher share one configuration-driven inference entry:

```bash
# Run the fixed SFT clean/fault case.
bash run.sh inference --config recipes/inference/cases.json
# Run the same fixed case with BF16 RL.
bash run.sh inference --config recipes/inference/rl_cases.json
# Run the unguided API teacher in evaluation mode.
bash run.sh inference --config recipes/inference/teacher_cases.json
# Run the API teacher with training-data guidance and export checks.
bash run.sh inference --config recipes/inference/teacher_data_cases.json
```

| Configuration | Model | Mode | Teacher guidance |
|---|---|---|---|
| `cases.json` | SFT | `evaluation` | None |
| `rl_cases.json` | BF16 RL | `evaluation` | None |
| `teacher_cases.json` | API teacher | `evaluation` | None |
| `teacher_data_cases.json` | API teacher | `teacher_data` | `teacher.md` |

`data/cases/train.json` fixes the task, tools, initial state, seed and fault. `data/cases/test.json` provides an ignored-filter OOD example. Point a local evaluation recipe to that file to try the held-out mechanism. Teacher-data mode defaults to the supplied 80% Train manifest and accepts another Train collection manifest through `teacher.pool`; Val/Test and the held-out fault remain excluded. See [collection manifests](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-choose-a-teacher-collection-manifest) for selection and fraction-generation CLI commands.

Clean and fault conditions start from the same database. Faults act at eligible calls; `fixed_call_indices` can additionally restrict their positions. Each condition runs once, so this case entry does not perform full-Test supplementation. All models use temperature 0, 50 turns and a Base-tokenizer 32K context / 2K output budget.

Teacher-data mode removes guidance and checks that successful candidates fit the student context before export. Candidates still need data review. Evaluation mode produces only evaluation outputs. This separation lets the same runner generate examples without leaking Test into training.

API inference uses the same key configuration. [Runtime setup](../backends/enterpriseops/README.md) · [Teacher collection CLI](https://www.github.com/haohuiliang/RobustOps/blob/main/data/README.md#user-content-collect-sft-demonstrations)
