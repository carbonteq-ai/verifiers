# CarbonTeq Verifiers distribution

Current unpublished SDK isolation increment (2026-10-04): startup and thread
configuration disable fourteen audited built-in features. Loaded-thread
`experimentalFeature/list` readback must confirm strict false values before
turn dispatch; the worker retains a versioned proof record. Twenty-four native
fixtures and seven real pinned-SDK/local-provider cases pass. Independent review
passes eight readback variants and 116 consumer backend cases. A fresh signed-in
consumer canary passes in 14.04 seconds, with all 889 source/test hashes unchanged
and both original action credits preserved. This qualifies the audited SDK0.160
transport, not all future built-ins or semantic accuracy. Worker identities
change truthfully; publication and dependency adoption remain open.

Actual schema forwarding confirmation (2026-10-04): the environment's fresh
signed-in protocol-3 probe completes with a schema-valid, source-bound answer,
zero decision errors and one reconciled response (18,852 input/376 output,
124 reported reasoning detail). All 743 native/environment source hashes remain
unchanged during that attempt. This is one positive summary integration case,
not broad semantic accuracy. Evidence lives in the consumer calibration plan's
immutable `summary-sdk-live-03` directory; earlier failed attempts are retained.

Local unpublished SDK increment (2026-10-04): the isolated qualification worker
forwards an optional exact JSON Schema object to the pinned SDK's public
`turn_start` as `outputSchema`. Invalid envelope types fail before credential
copy or client dispatch; the absent-schema solver path is unchanged. Root and
critic each pass fifteen forwarding/usage cases; the broader SDK subset passes
eighteen with seven explicitly opt-in unpaid cases skipped. Scoped Ruff passes.
SDK-installed Pyright still reports 83 existing notification-type diagnostics
at the payload fallback, outside the new forwarding lines; no whole-worker
typing or schema-adherence claim is made. Worker SHA is
`3178b4d8f15fc634f3cf9ec901c7670119f9c3e1dbbc161844e90acc5d9f2735`.
Actual schema-constrained model qualification and publication remain open.

Local unpublished validation increment (2026-10-04): Python-mode strict receipt
re-admission protects session retention, server emission, trace capture/export
and selected execution projection before JSON can normalize copied fields.
Exact sampled booleans, token masks and token IDs are required for projection.
These are evidence-admission repairs, with no changed domain values or trainer
advantages. Regression counts and current source hashes belong to the framework
`execution-token-alignment-checkpoint.md`; no immutable pin is updated here.

Status: independently maintained CarbonTeq distribution. Repository:
`https://github.com/carbonteq-ai/verifiers`. Prime Intellect Verifiers remains
an upstream source of reviewed changes, but upstream acceptance is not a release
or support requirement for this distribution.

Upstream synchronization base: primeintellect-ai/verifiers `main`,
`27bbd216df0af719a43705866b2cf6139bcc95de` (verified live 2026-09-08).
Release branch: `codex/carbonteq-verifiers-latest`.
Expected remotes: `origin=https://github.com/carbonteq-ai/verifiers.git` and
`upstream=https://github.com/PrimeIntellect-ai/verifiers.git`.

The 2026-09-08 synchronization advances one upstream commit from the previous
`e3bcbcbe` base. It restores direct Prime background-job polling in
`verifiers/v1/runtimes/prime.py`; this is upstream behavior, not a CarbonTeq
delta. Prime-RL main commit `04a61d3b` pins Verifiers `828488ff`, which is an
ancestor of this base. Its asynchronous orchestrator uses Verifiers'
consumer-stamped `TrainWorkInfo.policy: PolicySpan`; Verifiers deliberately
does not own trainer policy versions or staleness decisions.

## Async-training compatibility boundary

Verifiers is compatible with multiple asynchronous trainers through evidence
and lifecycle primitives; it is not the owner of their scheduling policy.
`TrainWorkInfo.policy: PolicySpan` records the oldest and newest live policy
versions spanned by an episode. Native traces retain exact sampled token IDs,
sampling masks, and behavior log probabilities. Those facts are sufficient for
a consumer to implement version rejection and importance-sampling correction.
The consumer's bounded queue and request admission implement depth bounding.

The current API supports these partial-rollout strategies:

- batch/request barriers, where no update occurs until active generation is
  complete;
- soft request drain, where the inference adapter stops accepting new model
  requests while existing requests finish, and an episode executing a tool
  remains alive until its next model request is admitted;
- acknowledged whole-episode or whole-group cancellation through caller-owned
  run IDs, without converting cancellation into a fabricated low reward.

The current `TrainClient` does not expose a partially completed assistant
generation. Therefore abort-and-prefix-resume and explicit mid-generation
save/resume are not yet generic Verifiers capabilities. A backend claiming one
of those modes must provide a generation adapter that retains partial token
IDs, aligned behavior log probabilities, sampling masks, and the same logical
turn/session identity across resumption. A completed tool call is environment
state and must never be replayed because weights changed. Mixed-weight
per-forward continuation is also unsupported until a provider supplies an
honest policy span and the consuming loss is qualified for that behavior.

Version rejection, queue depth, cancellation policy, importance-ratio math,
and weight transport remain trainer/orchestrator responsibilities. Do not add
those policies to task, scorer, or environment configuration.

## Maintained delta

The native environment-server wire contract carries both task data and the
task's validated per-instance config. Tasksets can derive row-specific config
while loading—for example, an allowlist of tools selected for one task—and a
worker must not silently replace that config with the catalog default when it
reconstructs the task. Older clients may omit `task_config` and retain the
static-config behavior. The CLI and CarbonTeq Posttrain caller send the complete
task. Regression coverage proves both the derived-config and legacy paths.

The training client depends on `carbonteq-renderers`, CarbonTeq's fork of the
`renderers` package (import name `renderers`; ledger in
`carbonteq-ai/renderers` `CARBONTEQ_FORK.md`). The fork owns every model output
format Posttrain trains, including LFM2.5's pythonic tool calls and tool-cycle
bridge and K2-Horizon's IFM formats, so Verifiers carries no model-specific
parsers. Each parse reports `reasoning_tokens`, the completion tokens that were
reasoning, and `response_from_generate` copies it into `Usage.reasoning_tokens`.
`pyproject.toml` lists PyPI and then the internal `carbonteq-dev` index (`https://pypi.lan/carbonteq/dev/+simple/`) for this repository's own lock, with no `carbonteq-renderers` source. uv applies a Git dependency's sources in every consumer, so a source pin here would force consumers outside the LAN to reach `pypi.lan`; instead each consumer supplies the published wheel from its own index or wheelhouse. `/inference/v1/generate` returns no usage details, so before this change the
reasoning share of a train-path reply was always unknown. The wire usage block
carries it as `completion_tokens_details.reasoning_tokens`. Regression:
`tests/v1/test_train_client.py::test_train_response_reports_the_renderer_reasoning_token_count`.

`Env.serving(client_factory=...)` accepts an optional host-owned client factory
and threads it through server, static-pool and elastic-pool interception.
The default remains the upstream client resolver. A host can supply an adapter
around an already loaded policy without launching a second inference engine.
Servers own adapter closure; factories should return a fresh adapter per
server/config. Closing a borrowed adapter must not unload its host's model.
Judge plugins retain upstream endpoint configuration and scoring ownership.

Changed files: `verifiers/v1/env.py`, `clients/client.py`,
`interception/__init__.py`, `interception/pool.py`,
`interception/server.py`, and existing `tests/v1/test_e2e.py`.
No algorithm, Posttrain package, or GPU-runtime dependency is introduced.

`TrainClientConfig.chat_template` optionally carries the exact selected Jinja
template to native environment workers. `ElasticRendererPool` applies it after
loading the tokenizer and includes it in the pool cache key, so two model
contracts cannot accidentally share a renderer merely because they use the
same base-model artifact. `None` preserves upstream behavior. This is required
when a training product versions a corrected template independently of the
model repository; it keeps worker-side rendering token-identical without
introducing Posttrain or task-specific logic into Verifiers.

`EnvClient.run(..., request_id=...)` optionally preserves a caller-stable wire
identity, and `EnvClient.cancel(request_id)` waits for the native server or pool
to acknowledge whether that run was still active. The existing automatic
best-effort cancel on coroutine cancellation remains a fallback. This lets a
fixed-policy coordinator prove episode termination before changing weights
without introducing trainer-specific identities or lifecycle code here.

`Taskset.select(keys)` and `EvalConfig.task_keys` allow an evaluation caller to
dispatch an exact ordered set of stable task identities. Selection materializes
only finite tasksets and rejects missing requested keys, duplicate requests,
and duplicate source identities before any episode starts. The ordinary
shuffle/head path remains unchanged when `task_keys` is absent. This generic
seam lets callers reuse a reviewed evaluation manifest across model subjects
without introducing Posttrain selection policy into Verifiers.

`EvalRunInfo.repetition_index` and the corresponding `RunSlot` field retain the
planned zero-based repetition identity on each standalone evaluation episode.
The eval runner stamps the identity before persisting the episode, so concurrent
completion order does not become an implicit repetition identifier. The field
is optional for compatibility with historical episode records.

Rollout startup and the local tool path are tuned for agentic RL, where every
episode starts its own tool server and harness program. `SubprocessConfig`
gains an opt-in fork server (`fork_server`, `preload`, defaulting to the
`VF_FORK_SERVER` and comma-separated `VF_FORK_SERVER_PRELOAD` environment
variables so one setting also reaches tool servers' own runtimes). One warm
interpreter per Python executable imports the listed modules once and forks
each `python script|-m module|-c code` program the runtime starts on that
interpreter, with the caller's session, working directory, environment and
stdio. Prepared uv script environments started through the runtime's own
activation wrapper (`UV_ACTIVATE_ARGV`) fork from a zygote for that
environment's interpreter, with the wrapper's variables applied. A `-m`
program's module is imported into the zygote on first use, so tool servers
start warm without being named in `preload`; if any import leaves a Python
thread running, the zygote retires. Anything else, or any fork-server
failure, falls back to exec. The pid
is reported only after the child's `setsid()` and the child is reaped only
after the caller has read its exit status, so process-group signals never reach
the zygote or a reused pid. Separately, a local server's port file is polled
with backoff instead of once a second, a subprocess-runtime server is probed
from the host instead of by a Python child process, and the MCP tool server
creates its listener with `IPPROTO_TCP` and `TCP_NODELAY`: asyncio enables
no-delay per connection only for `IPPROTO_TCP` sockets, and without it every
tool call waited about 40 ms on a delayed ACK. The same listener fix applies to
the Docker runtime's passed listener and NeMo Gym servers. For AutomationBench
at concurrency 16 these cut host time per episode from 6.6 s to 0.65 s.

Changed files: `verifiers/v1/runtimes/subprocess.py`, new
`verifiers/v1/runtimes/zygote.py` and `_zygote_server.py`,
`verifiers/v1/mcp/launch.py`, `verifiers/v1/mcp/server.py`,
`verifiers/v1/runtimes/docker/__init__.py`, and
`verifiers/v1/tasksets/nemo_gym/server.py`.

### Boxed-math scoring off the main thread

`verify_boxed_math_answer` bounds math-verify's `parse` and `verify` with
`parsing_timeout`/`timeout_seconds`, which math-verify enforces with
`signal.alarm`. Python allows that only on the main thread; off it, math-verify
raised and the broad `except` scored every answer 0.0, however correct.
Trainers that run episodes off the main thread (veRL scores inside Ray async
actors) therefore trained on zero rewards for every math-verify environment
(found by Posttrain qualification `q0412j-ws-verl-bf16-r1`, GSM8K: every trace
scored 0, including replies ending in the gold answer). Off the main thread the
pair is now scored in a one-process `spawn` worker pool, whose main thread
keeps the same timeouts; a worker that outlives its own alarm is killed and
replaced and the answer scores 0.0. Main-thread scoring is unchanged.
Regression: `test_boxed_math_answer_scores_off_the_main_thread` in
`tests/v1/test_scoring.py` fails before the change.

### Linked tool-server receipts record the submitted arguments

A linked MCP tool call (one carrying `verifiers.execution` dispatch metadata)
reaches the server's handler after argument validation has filled every
omitted optional parameter with its default. The receipt recorded those
expanded keyword arguments, so `validate_server_parent` rejected it as
"native server dispatch arguments changed" (HTTP 400) whenever the model left
an optional argument out, and the tool call failed before running. Posttrain
smoke run `manifest-steps-smoke-ws-20261005-r6` lost 54 of 57 AutomationBench
tool calls this way; the 3 that ran supplied every parameter. The metadata
middleware now keeps the submitted `arguments` for the linked request and the
receipt records those; unlinked capture is unchanged. Regression:
`test_real_mcp_reserved_metadata_capture_and_rejection_before_handler` in
`tests/v1/test_env_server.py` (its tool gains an omitted optional parameter)
fails before the change.

### Env-client replies decode inside a validation scope

`EnvClient` validated each episode reply outside any validation scope, so its
exact-match proof cache was off and every assessment batch re-validated the
same assessment source. AutomationBench episodes carry up to about 2,200
batches, and decoding took 2.6 CPU-s per episode on average (up to 51 s) on the
trainer's main process. The reply is now validated inside a per-reply
`validation_scope`, as the env server already does: the first occurrence is
fully validated, identical repeats reuse its proof, and the proofs end with the
reply. On 20 recorded Posttrain episodes the decoded episodes are identical and
decode CPU fell from 31.8 s to 2.1 s.

### Assessment archives load with reused intrinsic proofs

Episode and Trace validation own (or borrow) the bounded exact-content proof
scope through archive restoration and nested model validation, so a pooled
source or view that recurs across thousands of batches is fully validated once
per load. Contextual batch, invocation, provenance and credit checks still run
for every batch. Recorded AutomationBench archives with 2,736 and 60 batches
load in 2.08 s and 0.26 s instead of 258.08 s and 7.26 s, with identical
reserialized bytes; a scored DocuSign archive loads in 3.89 s instead of
84.01 s. Source: `verifiers/v1/episode.py`, `verifiers/v1/trace.py`
(`restore_assessment_archive` wrap validators). Regressions:
`test_scoring_archive_views_are_once_per_identity_with_exact_context` and the
intrinsic eviction/unicode tests in `tests/v1/test_scoring.py`.

### Assessment scoring, serialization and replies on large histories

AutomationBench manifest tasks retain hundreds of assessments; each attempt
appends queued, running, partial and terminal batches that refer to one large
source and view (about 250 KB each). Four changes remove repeated work without
changing any retained value:

- Env-server replies use the pooled archive form. `serve/server.py`
  (`_pack_response`) packs `model_dump(mode="python")` with the
  `ARCHIVE_CONTEXT` serialization context (`verifiers/v1/assessment_archive.py`),
  so each full source and view travels once and batches refer to it.
  `EnvClient` already restores the pooled form. Before, Python-mode dumps
  bypassed pooling: a 49.4 MB episode record became a 1,174.5 MB reply and a
  34.3 MB r6 record a 1,772.8 MB reply.
- Archive serialization (`serialize_archive`) elides repeated objects. Inside
  one encoding, a batch or credit request whose source or view object was
  already serialized emits a placeholder carrying the encoding's random token
  (`AssessmentBatch`/`CreditRequest` wrap field serializers,
  `verifiers/v1/assessments.py`). Normalization resolves only its own token's
  placeholders and re-serializes in full if any remains unresolved. Distinct
  objects with equal content are still serialized and compared in full, and
  filtered dumps are never elided. Serialization JSON schemas are unchanged.
- Intrinsic proofs (`verifiers/v1/_validation_scope.py`) keep strings in keys by
  reference, so large retained JSON is never re-encoded or re-hashed per lookup,
  and account large text per code point. This replaces the earlier UTF-8 key
  encoding with the same eviction bound. An exact object that already passed in
  scope and is deeply immutable (frozen models, tuples, scalars) is accepted by
  identity while its content proof is retained (`proven_instance`); a
  `model_copy` or re-parsed copy is checked on its own. Pydantic runs `after`
  validators even for instance inputs, so without this every lifecycle batch
  re-ran source and view verification.
- The plan admits source, requests and dependencies once and hands them to the
  attempt executor (`_execute_admitted`, `verifiers/v1/assessment_runtime.py`);
  lifecycle batches reuse those views and dependencies. Execution occurrence
  digests are cached by exact typed coordinates, and batch execution membership
  uses a hash index with tuple-scan fallback.

Evidence on 160 retained r6 episodes (Posttrain run
`manifest-steps-26-sampo-100-g16x8-20261005-r6`): `Episode.to_record` and
`Trace.to_record` output is byte-identical to 24c12379 for all 160, and the
episode decoded from the pooled reply reproduces the direct record for all 160.
Means fall from 280 to 142 ms (`Episode.model_validate_json` in a scope), 105 to
72 ms (`Episode.to_record`) and 104 to 64 ms (`Trace.to_record`); the reply
averages 7.4 MB, about the record size. On the ten heaviest (34–111 MB records,
984–3,644 batches; one process each) the means are: reply 2,322 MB to 58.5 MB,
pack 2.37 to 0.39 s, `EnvClient` decode 5.12 to 0.91 s, decode-process peak RSS
7.6 to 1.0 GB, `Episode.model_validate_json` 2.90 to 1.26 s, `Episode.to_record`
899 to 416 ms and `Trace.to_record` 964 to 352 ms.
Rescoring recorded 2.6B episodes with environment f587146 (48 from
`manifest-steps-26-sampo-100-g16x8-20261005-r3`, 111 from
`manifest-steps-26-g16x8-mean-20261005-r1`, plus the two heaviest manifest
episodes) gives identical findings, rewards, metrics, Posttrain turn rewards and
errors for all 161, ignoring only identifiers the environment draws from
`uuid4` per run. Unprofiled scoring falls from 5.86 to 3.00 s for
`marketing.email_blast_suppression` (551 assessments, 2,204 batches), 3.31 to
2.19 s for `support.gorgias_inventory_routing`, and 91.9 to 72.4 s over the
111-episode run. Retained records are unchanged by design.
Regressions in `tests/v1/test_scoring.py`:
`test_archive_serialization_elides_repeated_evidence_objects`,
`test_archive_serialization_still_rejects_conflicting_equal_identity_copies`,
`test_env_server_reply_pools_assessment_evidence_and_restores_identically`,
`test_proven_instances_are_exact_objects_bounded_by_retained_proofs`,
`test_batch_membership_index_matches_tuple_scan` and
`test_occurrence_digests_keep_copied_coordinate_types_apart`.
Implementation commits: `b50265738` (archive loading) and `d793cc6ee`
(scoring, serialization and replies) on `codex/native-assessment-runtime-cost`;
the consumer selection is recorded in Posttrain `docs/tooling/verifiers/README.md`.

### Compact archive references and lifecycle deltas

Pooling stored each source and view body once, but every batch still repeated
the full source identity (all node, trace and execution coordinates) and view
metadata (all subjects), about 32 KB per batch on AutomationBench; the four
lifecycle batches of each attempt also repeated its run and findings. In the
archive form (`verifiers/v1/assessment_archive.py`) a batch or credit request
now names its source as `{snapshot_id, episode_id}` and each inline view as
`archive_view_ref` with `view_id`, `snapshot_id`, `builder_revision`, `scope`
and `input_digest`. Both identities are content digests of the verified pooled
entries, so nothing is lost. An attempt's queued, running and progress batches
are kept, because task credit planners (AutomationBench
`manifest_assessments.plan_credit` checks lifecycle progression) and
calibration inspect them. Each is written as a delta against the attempt's
last batch (`archive_lifecycle_of`): run coordinates, differing run fields,
and assessments or receipts as a prefix length. Restoration rebuilds identical
in-memory batches. Older archives with full references still load: supplied
coordinates must equal the pooled entry's. Older Verifiers cannot read the new
form. Raw readers that keep the last batch per run key (Posttrain's Trackio
results projection) see the same results, since the last batch stays complete.

On the 100-episode AutomationBench eval `luna2-heldout-64k20t-t05-v2-final`
(902 MB of `write_episode` output) the same episodes write 140 MB, with no
episode over 10 MB (11 before). All 100 restore to identical batches and
assignments. The largest `support.zendesk_hubspot_org_sync` episode (3,936
batches) falls from 138.7 MB to 7.5 MB; loading takes 0.76 s instead of 2.21 s,
and writing 0.16 s instead of 0.65 s. Its env-server reply falls from 124.7 MB
to 6.8 MB. Regressions in `tests/v1/test_scoring.py`:
`test_scoring_archive_lifecycle_batches_roundtrip_as_compact_deltas` (JSON,
eval-writer and msgpack reply round trips plus forged deltas) and
`test_scoring_archive_loads_recorded_legacy_trace` (a trimmed recorded
AutomationBench trace, `tests/v1/fixtures/assessment_archive_legacy.json.gz`).

## Regression and compatibility

Use Python 3.13 and the selected upstream lock. The real local subprocess/null
harness test `test_host_client_factory_runs_local_episode_and_closes_adapter`
passes for server/static/elastic interception without external model credentials.

    uv run --python 3.13 python -m pytest tests/v1/test_e2e.py -k host_client_factory -q
    uv run --python 3.13 python -m pytest tests/v1 -q

The three injected-client cases and the current v1 suite pass; credentialed E2E
cases skip when `PRIME_API_KEY` is absent. Nine focused Posttrain train/eval
compatibility checks pass against the latest source, including exact
sampled-token/log-probability preservation and AutomationBench tool execution.
The selected-template config round trip and renderer-cache isolation tests pass
in `tests/v1/test_train_client.py`. The caller-owned run identity and
acknowledged cancellation contract passes in `tests/v1/test_e2e.py`.
Exact evaluation task selection passes `tests/v1/test_taskset.py` and
`tests/v1/test_eval_task_selection.py`; the complete v1 suite remains the
publication gate.
The fork server's exec parity (argv, `sys.path[0]`, cwd, environment,
`TMPDIR`, exit codes, pipes, merged background logs, signals, fallbacks and
64 concurrent programs) passes in `tests/v1/test_subprocess_fork_server.py`;
`tests/v1/test_mcp_server_latency.py` fails at about 42 ms per call without
the listener fix and passes (about 3 ms) with it.
Twenty-seven AutomationBench environment tests pass after removing its optional
OpenAI Agents schema dependency, which conflicts with Verifiers' MCP 2 runtime.
Consumer ownership and evidence are documented in Posttrain's
`docs/tooling/verifiers/README.md`.

## Upstream synchronization and publication

Upstream `main` at `27bbd216df0af719a43705866b2cf6139bcc95de` still has no
host-client injection seam. Periodically review upstream changes and port or
merge them by behavior, retaining CarbonTeq-owned APIs when they remain useful.
An upstream pull request may be opened when mutual reuse is valuable, but it is
never a promotion gate. Before synchronizing, inventory every CarbonTeq delta,
run fork and consumer compatibility suites, then build and clean-install the
wheel. Publish only immutable CarbonTeq commits and move consumer pins only
after qualification.

Published release: `carbonteq-v0.3.2.dev109` (GitHub release) at immutable
commit `bc70a7deaf64c8f8e0b41e39c00ac2d1ea1d7e0b` on
`codex/carbonteq-verifiers-latest` (a merge of `c4ba45e11` into the dev102
release line). Retained wheel SHA-256
`0b2bffc55471fbe3666c3b691822895e9a1bb77b7419893bfceb1b488916d16c`, source
distribution `ecd52fb894305e96249f8b0e13219ee5bbc4c5a32e82abeb88ab0d29f3254d91`.
It adds the eval-client transport retries (connect retry with backoff, 2 s
keep-alive expiry, retry of connections closed before any response, named empty
errors) and the compact archive references and lifecycle deltas described
above. Readers older than dev109 cannot read traces written in the compact form.
Consumer revision: Posttrain 0.4.16 and verifiers-environments
`carbonteq-2026.10.08` select this commit.

Previous release: `carbonteq-v0.3.2.dev102` (GitHub release, prerelease) at
immutable commit `74dd3fbf176d60cb0070f2408bc8a597bbfc099e` on `codex/carbonteq-verifiers-latest`
(`74dd3fbf1` is `58df1306` plus repository formatting, no behavior change).
Retained wheel SHA-256
`ce3bb4031f6a64566a9a552fed341cbbf7659f963d14b83643e909cff804489c`, source
distribution `4d48235af8620c9958493449a8518b3e838ac0188705c3af77eb6e41b8148e23`.
It includes the native assessment and credit runtime, submitted-argument
receipts, pooled assessment archives and scoped validation-proof reuse described
in the candidate sections below, which are now published.
Consumer revision: Posttrain 0.4.15 selected this commit.

## Local assessment source-admission candidate (2026-10-04)

The unpublished credit/assessment checkout adds
`AssessmentContext.retrospective_source()`. `execute_assessment` anchors the
already validated supplied snapshot in private invocation state. The accessor
freshly admits that snapshot and permits access only when all views are
retrospective; prefix/action-result contexts and calls outside native execution
cannot use it. The anchor is excluded from serialization, so native archives
remain the source authority and no new wire field or schema version is added.

Regression coverage is in `tests/v1/test_scoring.py`: actual native execution
supplies the genuine source even when a custom input view has different content;
prefix contexts cannot obtain future raw state. Selected scoring/trace/judges
tests and scoped Ruff pass; `assessments.py` and `assessment_runtime.py` pass
focused Pyright. Broader candidate/release qualification remains open.

AutomationBench uses this seam to authenticate deterministic record, guard,
occurrence and terminal-outcome input projections before publishing findings.
Consumer execution and remaining gates live in the RL assessment runbook and
`docs/tooling/verifiers/README.md`. This local work has not changed the published
implementation commit or an immutable consumer pin.

## Local intrinsic-validation reuse candidate (2026-10-04)

The private `_validation_scope.py` module permits execution-owned reuse of
successful `SourceSnapshot` and `ObservationView` intrinsic proofs. Keys retain
exact input strings and recursively type-sensitive metadata; copied boolean,
float or string coordinates cannot become integers through JSON readmission.
Runtime re-admission preserves Python types before validation. Task and Env
scoring own fresh scopes, standalone executors own or explicitly borrow a scope,
and only the executor's exact planned child task can share it. Exit and
cancellation close the owner and clear references. Proof storage has private
entry and accounted-byte bounds; this is not an RSS guarantee or persisted cache.

Source membership, current runs, parents, subject/prefix visibility, accepted
findings and credit recipients remain checked on every invocation. Archive
loading reuses proofs within its own load scope (see the maintained delta). No
public configuration, wire field, reward primitive or dependency was added.

The selected native suite passes 153 tests with no failures/skips, scoped
Ruff/Pyright pass, and independent critic's 30 focused intrinsic/archive-prefix
cases pass. The RL consumer's exact recorded content score/rescore/reload test
passes in 55.91 seconds versus the prior 195.00 seconds. Original scalar/source
bytes and exact credit/rescore/reload checks pass; this one local workload does
not establish training throughput or memory reduction. Whole-file hashes and remaining gates are retained in
`docs/research/verifiers-assessment-qualification/reward-candidate/native-intrinsic-proof-qualification.json`
in the RL repository. Candidate source remains unpublished and consumer pins
are unchanged.
Local unpublished increment, 2026-10-04: native MCP execution receipts can carry
host-issued dispatch tickets and separate transport-attempt coordinates. The
host validates original sampled call membership before server dispatch;
`resolve_execution_parent` exposes the retained dispatch prefix. Real local
Null/MCP retry and reload gates pass. Legacy unlinked receipts remain unlinked;
exact token projection and publication remain open. Consumer evidence is in
the RL checkout's `docs/research/verifiers-assessment-qualification/reward-candidate/native-parent-link-checkpoint.md`.

Local unpublished execution alignment increment, 2026-10-04: execution subjects
reuse exact generated-call projection through retained parent dispatches, keeping
execution identity and physical retry contributions separate. Strict current
receipt admission rejects copied boolean coordinates. Real-session coordinate
fixtures use manufactured exact parser evidence; production renderer and consumer
qualification remain separate. Consumer evidence:
`docs/research/verifiers-assessment-qualification/reward-candidate/execution-token-alignment-checkpoint.md`.

Published implementation commit: `265fccb9437eac0de212fb44b9bb425b1f9fd050`.
Consumer revision: Posttrain 0.4.5 selects `0cee0a075ddf1883498be0fde34155655cb19146`
(renderer reasoning-token counts through `carbonteq-renderers` 0.1.12.post1.dev1,
without a consumer-visible index pin).
