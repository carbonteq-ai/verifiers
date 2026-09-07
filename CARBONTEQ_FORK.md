# CarbonTeq Verifiers fork

Status: maintained CarbonTeq fork. Repository:
`https://github.com/carbonteq-ai/verifiers`. The immutable implementation and
consumer revisions are recorded below after publication.

Upstream: primeintellect-ai/verifiers, v0.3.1,
`b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`.
Local branch: `codex/injected-policy-client`.

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

Use Python 3.13 and the v0.3.1 upstream lock. The real local subprocess/null
harness test `test_host_client_factory_runs_local_episode_and_closes_adapter`
passes for server/static/elastic interception without external model credentials.

    uv run pytest tests/v1/test_e2e.py -k host_client_factory -q

Consumer tests additionally exercise exact sampled-token preservation and
AutomationBench tool execution. Consumer ownership and evidence are documented
in Posttrain's `docs/tooling/verifiers/README.md`.

## Rebase and publication

Upstream `main` at `e3bcbcbe5c55297a07a5d1038e37c2408b4a3dbd` still has no
host-client injection seam. Before rebasing, check again for an equivalent
upstream capability, run the three local integrations and the upstream
client/interception regression suites, then build and clean-install the wheel.

Published implementation commit:
`8e8f3042481c0996a58c3de0f86d55406725b6c1`.
Consumer revision: `8e8f3042481c0996a58c3de0f86d55406725b6c1`.
