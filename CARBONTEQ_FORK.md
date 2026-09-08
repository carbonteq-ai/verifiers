# CarbonTeq Verifiers distribution

Status: independently maintained CarbonTeq distribution. Repository:
`https://github.com/carbonteq-ai/verifiers`. Prime Intellect Verifiers remains
an upstream source of reviewed changes, but upstream acceptance is not a release
or support requirement for this distribution.

Upstream synchronization base: primeintellect-ai/verifiers `main`, 71 commits
after v0.3.1, `e3bcbcbe5c55297a07a5d1038e37c2408b4a3dbd`.
Release branch: `codex/carbonteq-verifiers-latest`.

## Maintained delta

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
Twenty-seven AutomationBench environment tests pass after removing its optional
OpenAI Agents schema dependency, which conflicts with Verifiers' MCP 2 runtime.
Consumer ownership and evidence are documented in Posttrain's
`docs/tooling/verifiers/README.md`.

## Upstream synchronization and publication

Upstream `main` at `e3bcbcbe5c55297a07a5d1038e37c2408b4a3dbd` still has no
host-client injection seam. Periodically review upstream changes and port or
merge them by behavior, retaining CarbonTeq-owned APIs when they remain useful.
An upstream pull request may be opened when mutual reuse is valuable, but it is
never a promotion gate. Before synchronizing, inventory every CarbonTeq delta,
run fork and consumer compatibility suites, then build and clean-install the
wheel. Publish only immutable CarbonTeq commits and move consumer pins only
after qualification.

Published implementation commit: `b126760eadbbbbfff6eb7badca845925ee20a885`.
Consumer revision: pending.
