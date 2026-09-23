# CarbonTeq Verifiers distribution

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

### Async-training compatibility boundary

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

The training client registers an `lfm2` structured-output parser for the
default renderer. It recognizes LFM2's special-token-delimited Python call
list using `ast.literal_eval`; sampled text is never executed. Its incremental
tool-cycle bridge also preserves the exact sampled token prefix when vLLM has
stripped the stop token, appending only the protocol close/newline scaffold and
new tool observations. Posttrain opts into this behavior through its versioned
LFM conversation contract. This is a generic model-protocol compatibility seam
and does not introduce task, environment, reward, or trainer-algorithm
ownership into Verifiers. The parser and bridge should move to the upstream
`renderers` package when that package accepts LFM2 as a native protocol.

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
environment's interpreter, with the wrapper's variables applied; anything
else, or any fork-server failure, falls back to exec. The pid
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

Published implementation commit: `b126760eadbbbbfff6eb7badca845925ee20a885`.
Consumer revision: pending.
