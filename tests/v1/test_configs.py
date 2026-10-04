"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig).
"""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,outputs,active,missing_first_usage,retry_first",
    [
        (100, (7, 5), False, False, False),
        (100, (7, 5), False, True, False),
        (100, (7, 5), False, False, True),
        (12, (7, 5), False, False, False),
        (12, (7, 5), True, False, False),
        (16384, (16382, 2), True, False, False),
        (16384, (16383, 2), True, False, False),
    ],
)
async def test_codex_sdk_actual_unpaid_usage_accounting(
    tmp_path, budget, outputs, active, missing_first_usage, retry_first
):
    """Exercise the pinned app-server decoder with synthetic provider counters."""
    import asyncio
    import json
    import os

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from verifiers.v1.dialects.responses import ResponsesDialect
    from verifiers.v1.harnesses.codex_sdk.harness import MODEL_CATALOG

    if os.environ.get("VF_RUN_CODEX_SDK_USAGE") != "1":
        pytest.skip("set VF_RUN_CODEX_SDK_USAGE=1 for unpaid pinned SDK accounting")
    calls = []
    release = asyncio.Event()

    async def respond(request):
        body = await request.json()
        calls.append(
            {
                "model": body.get("model"),
                "keys": sorted(body),
                "tools": body.get("tools"),
            }
        )
        if retry_first and len(calls) == 1:
            error = {
                "type": "response.failed",
                "response": {
                    "id": "response_usage_failed",
                    "status": "failed",
                    "error": {
                        "code": "server_error",
                        "message": "Synthetic retry fixture",
                    },
                    "usage": None,
                },
            }
            return web.Response(
                body=f"data: {json.dumps(error)}\n\n".encode(),
                content_type="text/event-stream",
            )
        ordinal = len(calls) - int(retry_first)
        assert ordinal <= 3
        if ordinal == 3:
            await release.wait()
            return web.json_response(
                {"error": {"message": "fixture shut down"}}, status=400
            )
        response = {
            "id": f"response_usage_{ordinal}",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna",
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "metadata": {},
            "parallel_tool_calls": False,
            "usage": {
                "input_tokens": 10,
                "output_tokens": outputs[ordinal - 1],
                "output_tokens_details": {"reasoning_tokens": 3 if ordinal == 1 else 2},
                "total_tokens": 10 + outputs[ordinal - 1],
            },
        }
        if ordinal == 1 and missing_first_usage:
            response["usage"] = None
        if ordinal == 1 or active:
            # An intentionally unregistered call produces a local tool error,
            # requiring a second model response without filesystem or paid tools.
            item = {
                "type": "function_call",
                "id": f"fc_usage_{ordinal}",
                "call_id": f"call_usage_{ordinal}",
                "name": "unregistered_usage_fixture",
                "arguments": "{}",
                "status": "completed",
            }
            response["output"] = [item]
            events = [
                {
                    "type": "response.created",
                    "response": {**response, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "arguments": "", "status": "in_progress"},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "delta": "{}",
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": 0,
                    "arguments": "{}",
                },
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            chunks = [
                f"data: {json.dumps({**event, 'sequence_number': i})}\n\n".encode()
                for i, event in enumerate(events)
            ]
        else:
            response["output"] = [
                {
                    "type": "message",
                    "id": "msg_usage",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Unpaid usage fixture complete.",
                            "annotations": [],
                        }
                    ],
                }
            ]
            chunks = ResponsesDialect().stream_events(response)
        return web.Response(body=b"".join(chunks), content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/responses", respond)
    async with TestServer(app) as server:
        worker = (
            Path(__file__).resolve().parents[2]
            / "verifiers/v1/harnesses/codex_sdk/worker.py"
        )
        request = {
            "qualification_endpoint": str(server.make_url("/v1")),
            "model": "gpt-6-luna",
            "model_catalog": MODEL_CATALOG,
            "input": [{"type": "text", "text": "Run the unpaid usage fixture."}],
            "system_prompt": "Synthetic accounting qualification.",
            "mcp_urls": {},
            "approved_mcp_tools": {},
            "tool_timeout": 10,
            "timeout": 60,
            "output_budget": budget,
        }
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "--no-config",
            "--script",
            str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate((json.dumps(request) + "\n").encode()), 90
            )
        except BaseException:
            process.kill()
            await process.wait()
            raise
        finally:
            release.set()
    records = [json.loads(line) for line in stdout.splitlines()]
    (tmp_path / "events.json").write_text(json.dumps(records, indent=2))
    (tmp_path / "provider-calls.json").write_text(json.dumps(calls, indent=2))
    assert process.returncode == 0, (records, stderr.decode()[-1000:])
    assert all(not call["tools"] for call in calls)
    from verifiers.v1.harnesses.codex_sdk.worker import (
        BUILTIN_TOOL_ISOLATION,
        DISABLED_BUILTIN_FEATURES,
    )

    isolation = [item for item in records if item["kind"] == "builtin_tool_isolation"]
    assert len(isolation) == 1
    assert isolation[0]["revision"] == BUILTIN_TOOL_ISOLATION
    assert isolation[0]["scope"] == "loaded_thread"
    assert isolation[0]["disabled_features"] == DISABLED_BUILTIN_FEATURES
    usages = [
        x["event"]["params"]["tokenUsage"]["total"]
        for x in records
        if x.get("event", {}).get("method") == "thread/tokenUsage/updated"
    ]
    assert len(calls) in ({2, 3} if active else {2 + int(retry_first)})
    reported_output = outputs[1] if missing_first_usage else sum(outputs)
    assert usages[-1]["outputTokens"] == reported_output
    assert usages[-1]["reasoningOutputTokens"] == (2 if missing_first_usage else 5)
    assert usages[-1]["totalTokens"] == usages[-1]["inputTokens"] + reported_output
    completed = [
        x["event"]["params"]
        for x in records
        if x.get("event", {}).get("method") == "rawResponse/completed"
    ]
    assert [x["responseId"] for x in completed] == [
        "response_usage_1",
        "response_usage_2",
    ]
    assert (
        completed[0]["usage"] is None
        if missing_first_usage
        else completed[0]["usage"]["outputTokens"] == outputs[0]
    )
    assert completed[1]["usage"]["outputTokens"] == outputs[1]
    assert completed[1]["usage"]["reasoningOutputTokens"] == 2
    interrupts = [x for x in records if x["kind"] == "budget_interrupt"]
    assert len(interrupts) == int(sum(outputs) >= budget)
    if interrupts:
        assert interrupts[0]["observed_usage"]["outputTokens"] == sum(outputs)
    assert usages[0]["outputTokens"] < budget
    assert records[-1]["kind"] == "finished"
    errors = [
        x["event"]["params"]
        for x in records
        if x.get("event", {}).get("method") == "error"
    ]
    if retry_first:
        assert errors and errors[0]["willRetry"] is True
        assert records[-1]["status"] == "completed"
        assert (
            records[-1]["output_budget_state"] == "unqualified_provider_errors_observed"
        )
        assert records[-1]["provider_retries_observed"] >= 1
    else:
        assert not errors


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


@pytest.mark.asyncio
async def test_codex_initial_messages_preserve_user_content_and_task() -> None:
    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
    from verifiers.v1.task import TaskData
    from verifiers.v1.types import SystemMessage, UserMessage

    data = TaskData(
        system_prompt="Explicit instructions",
        prompt=[
            SystemMessage(role="system", content="Environment instructions"),
            SystemMessage(
                role="system",
                content=[
                    {"type": "text", "text": "More "},
                    {"type": "text", "text": "instructions"},
                ],
            ),
            UserMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Solve this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            ),
            UserMessage(role="user", content="And this"),
        ],
    )
    before = data.model_dump_json()
    harness = CodexHarness(CodexHarnessConfig(id="codex"))
    harness.build_env = AsyncMock(return_value={"fixture": "environment"})
    config = await harness.prepare_acp(
        None, None, SimpleNamespace(), "endpoint", "secret", {}, data
    )
    assert (
        config.system_prompt
        == "Explicit instructions\n\nEnvironment instructions\n\nMore instructions"
    )
    assert config.prompt == data.prompt[2:]
    assert config.prompt[0] is data.prompt[2]
    assert config.env == {"fixture": "environment"}
    assert data.model_dump_json() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "Only system"}],
        [{"role": "assistant", "content": "History"}],
        [{"role": "tool", "content": "History", "tool_call_id": "call"}],
        [
            {"role": "user", "content": "Task"},
            {"role": "system", "content": "Late system"},
        ],
        [
            {
                "role": "system",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            },
            {"role": "user", "content": "Task"},
        ],
    ],
)
async def test_codex_initial_messages_reject_unsupported_roles(messages) -> None:
    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
    from verifiers.v1.task import TaskData

    data = TaskData(prompt=messages)
    before = data.model_dump_json()
    harness = CodexHarness(CodexHarnessConfig(id="codex"))
    harness.build_env = AsyncMock(return_value={})
    with pytest.raises(ValueError, match="Codex initial"):
        await harness.prepare_acp(
            None, None, SimpleNamespace(), "endpoint", "secret", {}, data
        )
    harness.build_env.assert_not_awaited()
    assert data.model_dump_json() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [None, "Instructions"])
async def test_codex_string_prompt_retains_existing_folding(system) -> None:
    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
    from verifiers.v1.task import TaskData

    data = TaskData(prompt="Task", system_prompt=system)
    before = data.model_dump_json()
    harness = CodexHarness(CodexHarnessConfig(id="codex"))
    harness.build_env = AsyncMock(return_value={})
    config = await harness.prepare_acp(
        None, None, SimpleNamespace(), "endpoint", "secret", {}, data
    )
    assert config.system_prompt is None
    assert config.prompt == ("Task" if system is None else "Instructions\n\nTask")
    assert data.model_dump_json() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [False, True])
async def test_docker_helper_images_configure_commands_and_runtime_record(
    monkeypatch, pinned
) -> None:
    import array
    import socket

    import verifiers.v1.runtimes.docker as module
    from verifiers.v1.runtimes.base import ProgramResult

    listener_image = "python:3.11-alpine"
    network_image = "alpine:3.22"
    if pinned:
        listener_image += (
            "@sha256:f2cdc43fcddbabe870f53750cbdcc01ae4aa75b1959351252457fde88f91d20f"
        )
        network_image += (
            "@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8"
        )
    values = {"allow": []}
    if pinned:
        values.update(listener_image=listener_image, network_setup_image=network_image)
    config = module.DockerConfig(**values)
    runtime = module.DockerRuntime(config)
    runtime._container = "fixture-container"
    runtime._proxy = SimpleNamespace(port=8123, policy=None)
    runtime._proxy_host_ip = "127.0.0.1"
    calls = []

    async def fake_docker(*args):
        calls.append(args)
        if listener_image in args:
            mount = args[args.index("--mount") + 1]
            directory = mount.split("source=", 1)[1].split(",", 1)[0]
            with socket.socket(socket.AF_UNIX) as control, socket.socket() as donor:
                control.connect(f"{directory}/control.sock")
                control.sendmsg(
                    [b"fd"],
                    [
                        (
                            socket.SOL_SOCKET,
                            socket.SCM_RIGHTS,
                            array.array("i", [donor.fileno()]),
                        )
                    ],
                )
        return ProgramResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(module, "docker", fake_docker)
    listener = await runtime._container_listener()
    listener.close()
    await runtime.prepare_execution(["http://localhost:1234/state"])
    listener_command, network_command = calls
    assert listener_command[listener_command.index(listener_image) + 1] == "python3"
    assert network_command[network_command.index(network_image) + 1] == "sh"
    for command, capability in (
        (listener_command, "DAC_OVERRIDE"),
        (network_command, "NET_ADMIN"),
    ):
        assert command[command.index("--network") + 1] == "container:fixture-container"
        assert command[command.index("--cap-drop") + 1] == "ALL"
        assert command[command.index("--cap-add") + 1] == capability
    assert (
        listener_command[listener_command.index("--security-opt") + 1]
        == "no-new-privileges"
    )
    assert "target=/run/vf" in listener_command[listener_command.index("--mount") + 1]
    assert module.DockerConfig.model_validate_json(config.model_dump_json()) == config
    restored = module.DockerRuntimeInfo.model_validate_json(
        runtime.info.model_dump_json()
    )
    assert restored.listener_image == listener_image
    assert restored.network_setup_image == network_image
    assert restored.image == "python:3.11-slim"


@pytest.mark.asyncio
async def test_codex_mcp_aliases_preserve_valid_names_and_namespace_consistency() -> (
    None
):
    import hashlib
    import json
    import re
    from collections import Counter

    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig

    empty_candidate = f"vf_mcp_{hashlib.sha256(b'').hexdigest()[:12]}"
    urls = {
        "": "http://fixture/empty?token=private",
        empty_candidate: "http://fixture/reserved",
        empty_candidate + "_1": "http://fixture/reserved-suffix",
        "invalid.name": "http://fixture/invalid",
        "valid-name": "http://fixture/hyphen",
        "valid_name": "http://fixture/underscore",
    }
    outputs = []
    for entries in (urls, dict(reversed(list(urls.items())))):
        runtime = SimpleNamespace(write=AsyncMock())
        trace = SimpleNamespace(id="fixture", info={"existing": "preserved"})
        harness = CodexHarness(CodexHarnessConfig(id="codex"))
        env = await harness.build_env(
            SimpleNamespace(model="fixture-model"),
            trace,
            runtime,
            "endpoint",
            "secret",
            entries,
        )
        written = runtime.write.await_args.args[1].decode()
        servers = tomllib.loads(written)["mcp_servers"]
        aliases = trace.info["codex_mcp_server_aliases"]
        assert aliases[""] == empty_candidate + "_2"
        assert aliases[empty_candidate] == empty_candidate
        assert aliases[empty_candidate + "_1"] == empty_candidate + "_1"
        assert aliases["valid-name"] == "valid-name"
        assert aliases["valid_name"] == "valid_name"
        assert trace.info["existing"] == "preserved"
        assert "private" not in json.dumps(trace.info)
        assert len(set(aliases.values())) == len(urls)
        assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in servers)
        assert {raw: servers[alias]["url"] for raw, alias in aliases.items()} == urls
        bases = {
            name: (base if base.startswith("mcp__") else f"mcp__{base}")
            for name in servers
            for base in (re.sub(r"[^a-zA-Z0-9_]", "_", name),)
        }
        counts = Counter(bases.values())
        expected = []
        for name, namespace in bases.items():
            if counts[namespace] > 1:
                suffix = hashlib.sha1(f"{name}\0{name}\0".encode()).hexdigest()[:12]
                namespace = (
                    f"{namespace[:-2]}_{suffix}__"
                    if namespace.endswith("__")
                    else f"{namespace}_{suffix}"
                )
            expected.append(namespace)
            if len(namespace) > 49:
                expected.append(namespace[:49])
        actual = json.loads(env["CODEX_CONFIG"])["features"]["code_mode"][
            "direct_only_tool_namespaces"
        ]
        assert actual == list(dict.fromkeys(expected))
        outputs.append((written, env["CODEX_CONFIG"], aliases))
    assert outputs[0] == outputs[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs", [(), (20, 30), (16384, 16385), (20, 10), (-1,)])
@pytest.mark.parametrize(
    "schema", [None, {"type": "object", "additionalProperties": False}, []]
)
async def test_codex_sdk_worker_noauth_empty_environments_and_private_events(
    monkeypatch, tmp_path, capsys, outputs, schema
) -> None:
    import json
    import os
    import sys
    from itertools import pairwise

    from verifiers.v1.harnesses.codex_sdk.harness import MODEL_CATALOG
    from verifiers.v1.harnesses.codex_sdk.worker import (
        BUILTIN_TOOL_ISOLATION,
        DISABLED_BUILTIN_FEATURES,
        run_qualification,
    )

    requests = []
    events = iter(
        [
            ("account/updated", {"authMarker": "private-auth-marker"}),
            (
                "item/completed",
                {
                    "threadId": "thread",
                    "turnId": "turn",
                    "item": {
                        "type": "agentMessage",
                        "id": "item",
                        "text": "Fixture complete",
                    },
                },
            ),
            *[
                (
                    "thread/tokenUsage/updated",
                    {
                        "threadId": "thread",
                        "turnId": "turn",
                        "tokenUsage": {"total": {"outputTokens": output}},
                    },
                )
                for output in outputs
            ],
            (
                "turn/completed",
                {"threadId": "thread", "turn": {"id": "turn", "status": "completed"}},
            ),
        ]
    )

    class Payload:
        def __init__(self, raw):
            self.raw = raw

        def model_dump(self, **kwargs):
            return self.raw

    class Client:
        def __init__(self, config):
            assert "OPENAI_API_KEY" not in os.environ
            assert "PRIVATE_SCORER" not in os.environ
            assert os.environ["HOME"] == str(tmp_path)
            assert config.experimental_api
            assert any(
                item.startswith("model_catalog_json=")
                for item in config.config_overrides
            )
            assert all(
                f"features.{name}=false" in config.config_overrides
                for name in DISABLED_BUILTIN_FEATURES
            )
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def initialize(self):
            return Payload({"version": "fixture"})

        async def request(self, method, params, **kwargs):
            requests.append((method, params))
            if method == "experimentalFeature/list":
                return SimpleNamespace(
                    root={
                        "data": [
                            {"name": name, "enabled": False}
                            for name in DISABLED_BUILTIN_FEATURES
                        ],
                        "nextCursor": None,
                    }
                )
            if method == "mcpServerStatus/list":
                return SimpleNamespace(
                    root={
                        "data": [
                            {
                                "name": "benchmark",
                                "tools": {"execute_tool": {}, "search_tools": {}},
                                "runtimeStatus": "connected",
                                "url": "private-controller-endpoint",
                                "auth": "private-inventory-auth",
                            }
                        ]
                    }
                )
            return SimpleNamespace(
                root={"thread": {"id": "thread", "environments": []}}
                if method == "thread/start"
                else {"turn": {"id": "turn"}}
            )

        async def turn_start(self, thread_id, inputs, params):
            requests.append(
                ("turn/start", {**params, "threadId": thread_id, "input": inputs})
            )
            response = Payload({"turn": {"id": "turn"}})
            response.turn = SimpleNamespace(id="turn")
            return response

        async def next_turn_notification(self, turn_id):
            assert turn_id == "turn"
            method, raw = next(events)
            return SimpleNamespace(method=method, payload=Payload(raw))

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        SimpleNamespace(
            CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai_codex.async_client",
        SimpleNamespace(AsyncCodexClient=Client),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai_codex.errors",
        SimpleNamespace(
            InvalidRequestError=type("InvalidRequestError", (Exception,), {})
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "private-auth-marker")
    monkeypatch.setenv("PRIVATE_SCORER", "private-scorer-marker")
    request = {
        "qualification_endpoint": "http://127.0.0.1:1234/v1",
        "model": "gpt-6-luna",
        "model_catalog": MODEL_CATALOG,
        "approved_mcp_tools": {"benchmark": ("execute_tool", "search_tools")},
        "mcp_urls": {"benchmark": "http://127.0.0.1:1235/mcp"},
        "tool_timeout": 10,
        "timeout": 10,
        "output_budget": 16384,
        "input": [{"type": "text", "text": "Public task"}],
        "system_prompt": "Public instructions",
    }
    if schema is not None:
        request["output_schema"] = schema
    if isinstance(schema, list):
        with pytest.raises(ValueError, match="output_schema"):
            await run_qualification(request, str(tmp_path))
        assert requests == []  # No client dispatch on malformed schema envelope.
        assert not (tmp_path / "codex").exists()
        return
    invalid = any(output < 0 for output in outputs) or any(
        right < left for left, right in pairwise(outputs)
    )
    if invalid:
        with pytest.raises(ValueError, match="counter"):
            await run_qualification(request, str(tmp_path))
    else:
        await run_qualification(request, str(tmp_path))
    initial, features, inventory, turn = requests[:4]
    assert initial[0] == "thread/start" and initial[1]["environments"] == []
    assert initial[1]["experimentalRawEvents"] is True
    assert initial[1]["ephemeral"] is True
    assert all(
        initial[1]["config"]["features"][name] is False
        for name in DISABLED_BUILTIN_FEATURES
    )
    assert features == (
        "experimentalFeature/list",
        {"threadId": "thread", "limit": 100, "cursor": None},
    )
    assert initial[1]["config"]["mcp_servers"]["benchmark"]["required"] is True
    assert initial[1]["config"]["mcp_servers"]["benchmark"]["tools"] == {
        "execute_tool": {"approval_mode": "approve"},
        "search_tools": {"approval_mode": "approve"},
    }
    assert (
        "default_tools_approval_mode"
        not in initial[1]["config"]["mcp_servers"]["benchmark"]
    )
    assert inventory == (
        "mcpServerStatus/list",
        {"threadId": "thread", "serverName": "benchmark", "detail": "full"},
    )
    assert turn[1]["environments"] == []
    if schema is None:
        assert "outputSchema" not in turn[1]
    else:
        assert turn[1]["outputSchema"] == schema
    assert "private-auth-marker" not in json.dumps(requests)
    assert "private-scorer-marker" not in json.dumps(requests)
    retained = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["kind"] for item in retained[:6]] == [
        "capability_catalog",
        "initialized",
        "thread_started",
        "builtin_tool_isolation",
        "mcp_inventory",
        "turn_started",
    ]
    assert retained[3] == {
        "kind": "builtin_tool_isolation",
        "revision": BUILTIN_TOOL_ISOLATION,
        "scope": "loaded_thread",
        "thread_id": "thread",
        "disabled_features": DISABLED_BUILTIN_FEATURES,
    }
    assert retained[4] == {
        "kind": "mcp_inventory",
        "scope": "thread_bound",
        "servers": [
            {
                "name": "benchmark",
                "tool_names": ["execute_tool", "search_tools"],
                "runtime_status": "connected",
            }
        ],
    }
    assert "private-controller-endpoint" not in json.dumps(retained)
    assert "private-inventory-auth" not in json.dumps(retained)
    catalog = json.loads(Path(initial[1]["config"]["model_catalog_json"]).read_text())
    original = MODEL_CATALOG["models"][0]
    narrowed = catalog["models"][0]
    assert narrowed == {
        **original,
        "tool_mode": "direct",
        "multi_agent_version": "disabled",
        "experimental_supported_tools": [],
    }
    assert original["tool_mode"] == "code_mode_only"
    assert retained[-1]["kind"] == ("interrupted" if invalid else "finished")
    if invalid:
        assert any(item["kind"] == "usage_unusable" for item in retained)
    else:
        assert retained[-1]["output_budget_state"] == (
            "observed_output_only" if outputs else "unavailable"
        )
    interrupts = [call for call in requests if call[0] == "turn/interrupt"]
    assert len(interrupts) == (
        1 if invalid or any(value >= 16384 for value in outputs) else 0
    )
    assert sum(item["kind"] == "budget_interrupt" for item in retained) == (
        1 if any(value >= 16384 for value in outputs) else 0
    )
    assert "private-auth-marker" not in json.dumps(retained)
    assert os.environ["OPENAI_API_KEY"] == "private-auth-marker"


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "enabled",
        "zero",
        "string",
        "duplicate",
        "missing_cursor",
        "cursor",
    ],
)
async def test_codex_sdk_loaded_feature_readback_fails_closed(fault, capsys):
    import json

    from verifiers.v1.harnesses.codex_sdk.worker import (
        DISABLED_BUILTIN_FEATURES,
        confirm_builtin_tool_isolation,
    )

    entries = [{"name": name, "enabled": False} for name in DISABLED_BUILTIN_FEATURES]
    if fault == "missing":
        entries.pop(0)
    elif fault in {"enabled", "zero", "string"}:
        entries[0]["enabled"] = {"enabled": True, "zero": 0, "string": "false"}[fault]
    elif fault == "duplicate":
        entries.append(entries[0])
    pages = [
        {"data": entries[:7], "nextCursor": "second"},
        {"data": entries[7:], "nextCursor": None},
    ]
    if fault == "missing_cursor":
        pages[1].pop("nextCursor")
    elif fault == "cursor":
        pages[1]["nextCursor"] = "second"
    calls = []

    class Client:
        async def request(self, method, params, **kwargs):
            assert method == "experimentalFeature/list"
            assert params["threadId"] == "loaded-thread"
            calls.append(params)
            return SimpleNamespace(root=pages[len(calls) - 1])

    if fault:
        with pytest.raises(ValueError, match="SDK builtin"):
            await confirm_builtin_tool_isolation(Client(), "loaded-thread", None)
        assert capsys.readouterr().out == ""
    else:
        await confirm_builtin_tool_isolation(Client(), "loaded-thread", None)
        assert [call["cursor"] for call in calls] == [None, "second"]
        records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(records) == 1
        assert records[0]["disabled_features"] == DISABLED_BUILTIN_FEATURES
        assert records[0]["thread_id"] == "loaded-thread"


def test_codex_sdk_refuses_real_provider_and_preserves_public_task():
    from pydantic import ValidationError

    from verifiers.v1.errors import HarnessError
    from verifiers.v1.harnesses.codex_sdk import CodexSdkHarness, CodexSdkHarnessConfig
    from verifiers.v1.harnesses.codex_sdk.harness import public_prompt
    from verifiers.v1.harnesses.codex_sdk.worker import controller_config
    from verifiers.v1.runtimes import SubprocessConfig
    from verifiers.v1.task import Task, TaskData
    from verifiers.v1.utils.compile import validate_pairing

    data = TaskData(
        system_prompt="Explicit",
        prompt=[
            {"role": "system", "content": "Public environment"},
            {"role": "user", "content": "Public task"},
        ],
    )
    before = data.model_dump_json()
    assert public_prompt(data) == (
        "Explicit\n\nPublic environment",
        [{"type": "text", "text": "Public task"}],
    )
    assert data.model_dump_json() == before
    with pytest.raises(ValueError, match="loopback"):
        controller_config(
            {"qualification_endpoint": "https://api.openai.com/v1"}, "/tmp/fixture"
        )
    with pytest.raises(ValidationError, match="exactly one"):
        CodexSdkHarnessConfig(id="sdk")
    with pytest.raises(ValidationError, match="exactly one"):
        CodexSdkHarnessConfig(
            id="sdk",
            auth_file="/private/controller/auth.json",
            qualification_endpoint="http://localhost:1234/v1",
        )
    assert (
        CodexSdkHarnessConfig(
            id="sdk", auth_file="/private/controller/auth.json"
        ).qualification_endpoint
        is None
    )
    harness = CodexSdkHarness(
        CodexSdkHarnessConfig(id="sdk", auth_file="/private/controller/auth.json")
    )
    assert harness.EXECUTES_CODE is False
    validate_pairing(harness, Task, SubprocessConfig(), tools=("benchmark",))

    class ContainerTask(Task):
        NEEDS_CONTAINER = True

    with pytest.raises(ValueError, match="ContainerTask needs a container"):
        validate_pairing(harness, ContainerTask, SubprocessConfig())
    with pytest.raises(HarnessError, match="history"):
        public_prompt(
            TaskData(prompt=[{"role": "assistant", "content": "Previous answer"}])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [False, True])
async def test_codex_sdk_session_retains_partial_events_without_model_call_claims(
    success,
) -> None:
    import json

    from verifiers.v1.errors import HarnessError
    from verifiers.v1.harnesses.codex_sdk import CodexSdkHarness, CodexSdkHarnessConfig
    from verifiers.v1.task import WireTaskData

    records = [
        {
            "kind": "thread_started",
            "response": {"thread": {"id": "thread", "environments": []}},
        },
        {
            "kind": "event",
            "event": {
                "method": "item/completed",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn",
                    "item": {
                        "id": "item",
                        "type": "agentMessage",
                        "text": "Observed reply",
                    },
                },
            },
        },
        {
            "kind": "finished",
            "ok": success,
            "status": "completed" if success else "failed",
        },
    ]

    async def output():
        for record in records:
            yield (json.dumps(record) + "\n").encode()

    async def empty():
        if False:
            yield b""

    process = SimpleNamespace(
        stdout=output(),
        stderr=empty(),
        write=AsyncMock(),
        wait=AsyncMock(return_value=0 if success else 1),
        terminate=AsyncMock(),
        kill=AsyncMock(),
    )
    runtime = SimpleNamespace(
        supports_live_processes=True,
        prepare_uv_script=AsyncMock(return_value=["fixture-worker"]),
        open_process=AsyncMock(return_value=process),
    )
    harness = CodexSdkHarness(
        CodexSdkHarnessConfig(
            id="codex-sdk", qualification_endpoint="http://127.0.0.1:1234/v1"
        )
    )
    trace = SimpleNamespace(
        info={},
        root_reply=None,
        nodes=[],
        calls=[],
        state=SimpleNamespace(artifacts={}),
    )
    data = WireTaskData(prompt="Public task", hidden_assertions="private-scorer-marker")
    session = await harness.session(
        SimpleNamespace(model="fixture"),
        trace,
        runtime,
        "unused-model-endpoint",
        "private-auth-marker",
        {"": "http://localhost:1235/mcp"},
        data,
    )
    if success:
        result = await session._run(None)
        assert result.stdout == "Observed reply"
    else:
        with pytest.raises(HarnessError, match="partial events"):
            await session._run(None)
    retained = trace.info["codex_sdk"]
    assert retained["events"] == records
    assert json.loads(trace.state.artifacts["codex_sdk/events.json"]) == retained
    assert retained["fresh_thread_requested"] is True
    assert retained["output_budget"] == 16384
    assert retained["token_alignment"] == "unavailable"
    assert trace.nodes == [] and trace.calls == []
    request = json.loads(process.write.await_args.args[0])
    assert "private-scorer-marker" not in json.dumps(request)
    assert "private-auth-marker" not in json.dumps(request)
    process.terminate.assert_awaited_once()
    with pytest.raises(HarnessError, match="overwrite"):
        await session._run(None)
    assert trace.info["codex_sdk"] is retained
