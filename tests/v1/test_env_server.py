import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ConfigDict

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.task import Task, TaskData


class DerivedData(TaskData):
    marker: str


class DerivedConfig(TaskConfig):
    model_config = ConfigDict(frozen=True)

    allowed_tools: tuple[str, ...] = ()


class DerivedTask(Task[DerivedData, dict, DerivedConfig]):
    pass


def _server() -> EnvServer:
    server = EnvServer.__new__(EnvServer)
    server.data_cls = DerivedData
    server.task_cls = DerivedTask
    server.env = SimpleNamespace(
        config=SimpleNamespace(
            taskset=SimpleNamespace(
                task=DerivedConfig(allowed_tools=("catalog-default",))
            )
        )
    )
    return server


def test_env_server_preserves_per_task_derived_config() -> None:
    task = _server()._build_task(
        {"marker": "row-1"},
        {"allowed_tools": ["row-specific-tool"]},
    )

    assert task.data.marker == "row-1"
    assert task.config.allowed_tools == ("row-specific-tool",)


def test_env_server_accepts_legacy_data_only_request() -> None:
    task = _server()._build_task({"marker": "row-1"})

    assert task.config.allowed_tools == ("catalog-default",)


@pytest.mark.parametrize(
    "change",
    [
        {"transport_attempt_index": True},
        {"transport_attempt_index": "0"},
        {"transport_attempt_index": -1},
        {"parent_execution_id": ""},
        {"dispatch_ticket": 4},
        {"server_name": "spoofed"},
    ],
)
def test_dispatch_metadata_is_strict_and_server_namespace_is_not_client_input(change):
    from verifiers.v1.mcp.execution import DispatchMetadata

    with pytest.raises(ValueError):
        DispatchMetadata.model_validate(
            {
                "dispatch_ticket": "ticket",
                "parent_execution_id": "execution",
                "transport_attempt_index": 0,
                **change,
            }
        )


@pytest.mark.parametrize(
    "missing",
    [
        "parent_execution_id",
        "dispatch_ticket",
        "transport_attempt_index",
        "server_name",
    ],
)
def test_linked_receipt_requires_whole_quartet_and_legacy_wire_stays_absent(missing):
    from verifiers.v1.mcp.execution import ToolServerReceipt

    base = {
        "invocation_id": "call",
        "event_index": 0,
        "phase": "dispatch",
        "tool_name": "tool",
        "arguments_json": "{}",
    }
    legacy = ToolServerReceipt(**base).model_dump(mode="json")
    assert (
        not {
            "parent_execution_id",
            "dispatch_ticket",
            "transport_attempt_index",
            "server_name",
        }
        & legacy.keys()
    )
    link = {
        "parent_execution_id": "execution",
        "dispatch_ticket": "ticket",
        "transport_attempt_index": 0,
        "server_name": "",
    }
    assert ToolServerReceipt(**base, **link).server_name == ""
    link.pop(missing)
    with pytest.raises(ValueError, match="complete dispatch provenance"):
        ToolServerReceipt(**base, **link)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_execution_id", b"parent"),
        ("dispatch_ticket", b"ticket"),
        ("server_name", b"server"),
        ("transport_attempt_index", True),
    ],
)
async def test_server_receipt_emission_revalidates_raw_forged_link_types(field, value):
    import verifiers.v1 as vf

    class Tools(vf.Toolset[vf.ToolsetConfig, vf.State]):
        pass

    receipt = vf.ToolServerReceipt(
        invocation_id="invocation",
        event_index=0,
        phase="dispatch",
        tool_name="tool",
        arguments_json="{}",
        parent_execution_id="parent",
        dispatch_ticket="ticket",
        transport_attempt_index=0,
        server_name="server",
    )
    forged = receipt.model_copy(update={field: value})
    with pytest.raises(ValueError, match="valid (string|integer)"):
        await Tools(vf.ToolsetConfig())._emit_execution_receipt(forged)


@pytest.mark.asyncio
async def test_real_mcp_reserved_metadata_capture_and_rejection_before_handler():
    import httpx
    from mcp.server.mcpserver import MCPServer

    import verifiers.v1 as vf
    from verifiers.v1.mcp.server import (
        _execution_metadata_middleware,
        _request_execution,
    )

    receipts = []
    entered = []

    class LinkedTools(vf.Toolset[vf.ToolsetConfig, vf.State]):
        CAPTURE_EXECUTIONS = True
        TOOL_PREFIX = "actual_namespace"

        async def _emit_execution_receipt(self, receipt):
            # This seam represents the host ACK; ticket authorization itself is
            # exercised by the session/e2e tests, not invented by this server.
            if receipt.dispatch_ticket == "rejected":
                raise ValueError("host rejected dispatch ticket")
            receipts.append(receipt)

        @vf.tool
        def operation(self, value: str, suffix: str = "") -> str:
            entered.append(value)
            return value + suffix

    tools = LinkedTools(vf.ToolsetConfig())
    mcp = MCPServer("fixture", middleware=[_execution_metadata_middleware])
    tools.register(mcp)
    app = mcp.streamable_http_app(json_response=True, stateless_http=True)
    async with (
        mcp.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000"
        ) as client,
    ):

        async def call(metadata):
            params = {"name": "operation", "arguments": {"value": "observed"}}
            if metadata is not None:
                params["_meta"] = {"verifiers.execution": metadata}
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": params,
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-06-18",
                },
            )
            return response.json()

        metadata = {
            "parent_execution_id": "parent",
            "dispatch_ticket": "ticket",
            "transport_attempt_index": 2,
        }
        result = await call(metadata)
        assert not result["result"].get("isError", False)
        assert len(receipts) == 2 and entered == ["observed"]
        for receipt in receipts:
            assert receipt.parent_execution_id == "parent"
            assert receipt.dispatch_ticket == "ticket"
            assert receipt.transport_attempt_index == 2
            assert receipt.server_name == "actual_namespace"
            assert json.loads(receipt.arguments_json) == {
                "args": [],
                "kwargs": {"value": "observed"},
            }
        assert receipts[0].invocation_id == receipts[1].invocation_id
        assert _request_execution.get() is None
        for invalid in (
            {**metadata, "transport_attempt_index": True},
            {**metadata, "dispatch_ticket": "rejected"},
            {},
        ):
            result = await call(invalid)
            assert "error" in result or result["result"].get("isError")
            assert len(receipts) == 2 and entered == ["observed"]
            assert _request_execution.get() is None
        await call(None)
        assert len(receipts) == 4 and len(entered) == 2
        assert receipts[-1].parent_execution_id is None


@pytest.mark.asyncio
async def test_reserved_metadata_context_resets_on_cancellation_and_isolates_calls():
    from verifiers.v1.mcp.server import (
        _execution_metadata_middleware,
        _request_execution,
    )

    arrived = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def next_handler(ctx):
        seen.append(_request_execution.get().parent_execution_id)
        arrived.set()
        await release.wait()

    async def invoke():
        ctx = SimpleNamespace(
            method="tools/call",
            params={
                "_meta": {
                    "verifiers.execution": {
                        "parent_execution_id": "cancelled",
                        "dispatch_ticket": "ticket",
                        "transport_attempt_index": 0,
                    }
                }
            },
        )
        try:
            await _execution_metadata_middleware(ctx, next_handler)
        finally:
            assert _request_execution.get() is None

    task = asyncio.create_task(invoke())
    await arrived.wait()
    assert _request_execution.get() is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == ["cancelled"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raised", "interrupted", "capture_disabled"])
async def test_linked_server_failure_retains_provenance_and_resets_context(failure):
    import verifiers.v1 as vf
    from verifiers.v1.mcp.execution import active_execution, active_revision
    from verifiers.v1.mcp.server import (
        _execution_metadata_middleware,
        _request_execution,
    )

    receipts = []
    entered = []

    class LinkedTools(vf.Toolset[vf.ToolsetConfig, vf.State]):
        CAPTURE_EXECUTIONS = failure != "capture_disabled"
        TOOL_PREFIX = None

        async def _emit_execution_receipt(self, receipt):
            receipts.append(receipt)

    tools = LinkedTools(vf.ToolsetConfig())

    async def operation():
        entered.append(True)
        if failure == "interrupted":
            raise asyncio.CancelledError("cancelled operation")
        raise RuntimeError("failed operation")

    async def call_next(ctx):
        return await tools._with_state(operation)()

    ctx = SimpleNamespace(
        method="tools/call",
        params={
            "_meta": {
                "verifiers.execution": {
                    "parent_execution_id": "parent",
                    "dispatch_ticket": "ticket",
                    "transport_attempt_index": 0,
                }
            }
        },
    )
    expected = (
        asyncio.CancelledError
        if failure == "interrupted"
        else ValueError
        if failure == "capture_disabled"
        else RuntimeError
    )
    with pytest.raises(expected):
        await _execution_metadata_middleware(ctx, call_next)
    assert active_execution.get() is None
    assert active_revision.get() is None
    assert _request_execution.get() is None
    if failure == "capture_disabled":
        assert not receipts and not entered
    else:
        assert [receipt.phase for receipt in receipts] == ["dispatch", failure]
        assert receipts[0].invocation_id == receipts[1].invocation_id
        assert all(
            receipt.parent_execution_id == "parent"
            and receipt.server_name == ""
            and receipt.dispatch_ticket == "ticket"
            and receipt.transport_attempt_index == 0
            for receipt in receipts
        )


async def _admitted_linked_server_fixture():
    """Build the parent using real session admission, never a fabricated ticket."""
    import verifiers.v1 as vf
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.interception.tool import MCPDispatch, ToolHookRequest
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage

    call = ToolCall(id="provider-call", name="counter_bump", arguments="{}")
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="count")),
        nodes=[
            MessageNode(
                message=AssistantMessage(content="", tool_calls=[call]), sampled=True
            )
        ],
    )
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    message = ToolMessage(tool_call_id=call.id, content="", name=call.name)
    for index, phase in enumerate(("before", "dispatch")):
        hook = ToolHookRequest(
            phase=phase,
            message=message,
            execution_id="parent",
            call=call,
            event_index=index,
            mcp_dispatch=MCPDispatch(
                server_name="counter", tool_name="bump", arguments_json="{}"
            )
            if phase == "dispatch"
            else None,
        )
        decision = await session.handle_tool(phase, message, request=hook)
    receipt = vf.ToolServerReceipt(
        invocation_id="physical-invocation",
        event_index=0,
        phase="dispatch",
        tool_name="bump",
        arguments_json='{"args":[],"kwargs":{}}',
        parent_execution_id="parent",
        dispatch_ticket=decision["mcp_dispatch_ticket"],
        transport_attempt_index=0,
        server_name="counter",
    )
    return trace, session, receipt


@pytest.mark.asyncio
async def test_session_rejects_copied_boolean_transport_attempt_before_admission():
    trace, session, receipt = await _admitted_linked_server_fixture()
    original = trace.tool_execution_events
    forged = receipt.model_copy(update={"transport_attempt_index": True})
    with pytest.raises(ValueError, match="valid integer"):
        session.retain_tool_server_receipt(forged)
    assert trace.tool_execution_events == original
    assert trace.tool_state_revision == 0
    accepted = session.retain_tool_server_receipt(receipt)
    assert accepted["ok"] is True
    assert len(trace.tool_execution_events) == len(original) + 1
    assert session.retain_tool_server_receipt(receipt) == accepted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_tools",
        "empty_tools",
        "missing_role",
        "wrong_role",
        "duplicate_id",
        "missing_generated_attempt",
        "unsampled",
        "wrong_arguments",
    ],
)
async def test_source_parent_rejects_malformed_original_sampled_call(corruption):
    import copy

    from verifiers.v1.assessment_source import capture_trace_source, execution_refs

    trace, session, receipt = await _admitted_linked_server_fixture()
    session.retain_tool_server_receipt(receipt)
    source = capture_trace_source(trace)
    material = json.loads(source.source_json)
    assert (
        len(execution_refs(material, episode_id=trace.episode_id, trace_id=trace.id))
        == 2
    )
    forged = copy.deepcopy(material)
    node = forged["nodes"][0]
    message = node["message"]
    if corruption == "missing_tools":
        message.pop("tool_calls")
    elif corruption == "empty_tools":
        message["tool_calls"] = []
    elif corruption == "missing_role":
        message.pop("role")
    elif corruption == "wrong_role":
        message["role"] = "user"
    elif corruption == "duplicate_id":
        message["tool_calls"].append(copy.deepcopy(message["tool_calls"][0]))
    elif corruption == "missing_generated_attempt":
        for event in forged["tool_execution_events"]:
            if event["source"] == "harness":
                event["generated_attempt_index"] = 0
    elif corruption == "unsampled":
        node["sampled"] = False
    else:
        message["tool_calls"][0]["arguments"] = '{"different":true}'
    with pytest.raises((ValueError, TypeError)):
        execution_refs(forged, episode_id=trace.episode_id, trace_id=trace.id)
    assert json.loads(source.source_json) == material


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "coordinate,value",
    [
        ("attempt_index", True),
        ("attempt_index", "1"),
        ("attempt_index", 1.0),
        ("emitted_call_index", False),
        ("emitted_call_index", "0"),
        ("emitted_call_index", 0.0),
        ("unchanged", None),
    ],
)
async def test_source_parent_rejects_coerced_raw_generated_coordinates(
    coordinate, value
):
    import copy

    from verifiers.v1.assessment_source import capture_trace_source, execution_refs
    from verifiers.v1.types import GeneratedCallAttempt

    trace, session, receipt = await _admitted_linked_server_fixture()
    session.retain_tool_server_receipt(receipt)
    material = json.loads(capture_trace_source(trace).source_json)
    attempt = GeneratedCallAttempt(
        attempt_index=1,
        emitted_call_index=0,
        provider_call_id="provider-call",
        parse_status="parsed",
        raw="counter_bump({})",
        name="counter_bump",
        arguments="{}",
        coordinate_system="node_local_full_tokens",
        parser_revision="fixture-v1",
        span_fidelity="unavailable",
        completion_token_digest="fixture",
    )
    un_emitted = attempt.model_copy(
        update={
            "attempt_index": 0,
            "emitted_call_index": None,
            "provider_call_id": None,
            "parse_status": "malformed",
        }
    )
    material["nodes"][0]["generated_calls"] = [
        un_emitted.model_dump(mode="json"),
        attempt.model_dump(mode="json"),
    ]
    for event in material["tool_execution_events"]:
        if event["source"] == "harness":
            event["generated_attempt_index"] = 1
    # The original exact integer relation is admissible at this raw-source boundary.
    assert (
        len(execution_refs(material, episode_id=trace.episode_id, trace_id=trace.id))
        == 2
    )
    if coordinate == "unchanged":
        return
    forged = copy.deepcopy(material)
    forged["nodes"][0]["generated_calls"][1][coordinate] = value
    with pytest.raises(ValueError, match="generated-attempt coordinates"):
        execution_refs(forged, episode_id=trace.episode_id, trace_id=trace.id)


def test_tool_receipt_metadata_and_legacy_empty_state_authority():
    import verifiers.v1 as vf
    from verifiers.v1.assessment_source import capture_trace_source

    base = {"invocation_id": "invocation", "tool_name": "tool", "arguments_json": "{}"}
    for invalid in (
        {"phase": "dispatch", "event_index": 0, "state_write_revision": 1},
        {"phase": "dispatch", "event_index": 0, "state_conflict": False},
        {"phase": "dispatch", "event_index": 0, "state_error_json": '"error"'},
        {
            "phase": "raised",
            "event_index": 1,
            "error_json": '"error"',
            "state_persistence": "applied",
            "state_write_revision": 1,
        },
        {
            "phase": "interrupted",
            "event_index": 1,
            "error_json": '"error"',
            "state_conflict": True,
        },
        {
            "phase": "returned",
            "event_index": 1,
            "result_json": '"result"',
            "state_persistence": "applied",
        },
    ):
        with pytest.raises(ValueError):
            vf.ToolServerReceipt.model_validate({**base, **invalid})
    legacy = vf.ToolServerReceipt.model_validate(
        {
            **base,
            "phase": "returned",
            "event_index": 1,
            "result_json": '"result"',
            "state_persistence": "unknown",
        }
    )
    assert legacy.state_write_revision is None
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="task")),
    )
    source = capture_trace_source(trace)
    assert "tool_state_revision" not in json.loads(source.source_json)
    assert "state_write_receipts" not in json.loads(source.source_json)
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert restored.tool_state_revision == 0 and restored.state_write_receipts == ()
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id=trace.id,
    )
    assert vf.project_subject(subject, source, restored).status == "unsupported"
    restored.tool_state_revision = 1
    assert (
        vf.project_subject(subject, source, restored).reason
        == "current_tool_state_revision_changed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool", "cancel", "put"])
async def test_secondary_receipt_failure_preserves_original_primary_exception(failure):
    import httpx

    import verifiers.v1 as vf
    from verifiers.v1.mcp.execution import MAX_EXECUTION_RECEIPT_BYTES

    class CounterState(vf.State):
        value: int = 0

    primary = (
        RuntimeError("primary tool failure")
        if failure == "tool"
        else asyncio.CancelledError("primary cancellation")
        if failure == "cancel"
        else httpx.ConnectError("primary PUT network failure")
    )

    class BrokenReceiptTools(vf.Toolset[vf.ToolsetConfig, CounterState]):
        CAPTURE_EXECUTIONS = True

        async def _pull_state(self):
            return CounterState()

        async def _push_state(self, before):
            raise primary

        async def _emit_execution_receipt(self, receipt):
            if receipt.phase == "dispatch":
                return
            if failure == "put":
                # Production serialization/bound checks fail before attempting POST.
                return await super()._emit_execution_receipt(receipt)
            request = httpx.Request("POST", "http://receipt/tool-execution")
            raise httpx.HTTPStatusError(
                "secondary receipt rejected",
                request=request,
                response=httpx.Response(413, request=request),
            )

    tools = BrokenReceiptTools(vf.ToolsetConfig())

    def operation():
        tools.state.value = 1
        if failure in ("tool", "cancel"):
            raise primary
        return "x" * MAX_EXECUTION_RECEIPT_BYTES

    with pytest.raises(type(primary)) as caught:
        await tools._with_state(operation)()
    assert caught.value is primary
    assert any(
        "Secondary tool-execution capture failure" in note for note in primary.__notes__
    )
    assert tools.state.value == 0
    assert vf.record_execution_evidence({"outside": "call"}) is False


@pytest.mark.asyncio
async def test_real_mcp_receipts_survive_failure_and_expose_concurrent_state_conflicts(
    monkeypatch,
):
    from contextlib import AsyncExitStack

    import httpx
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from mcp.server.mcpserver import MCPServer

    import verifiers.v1 as vf
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession

    class ReceiptState(vf.State):
        value: int = 0

    class ReceiptTools(vf.Toolset[vf.ToolsetConfig, ReceiptState]):
        CAPTURE_EXECUTIONS = True
        TOOL_PREFIX = None

        @vf.tool
        async def change(
            self,
            value: int,
            fail: bool = False,
            parallel: bool = False,
            hold: bool = False,
        ) -> str:
            original = self.state.value
            self.state.value = value
            assert vf.record_execution_evidence(
                {"local_before": original, "local_after": value}
            )
            if hold:
                self.entered.set()
                await self.hold.wait()
            if parallel:
                self.arrived += 1
                if self.arrived == 2:
                    self.barrier.set()
                await self.barrier.wait()
            if fail:
                raise RuntimeError("tool failed after local mutation")
            return f"value:{value}"

    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="change")),
        state=ReceiptState(),
    )
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    interception = InterceptionServer.__new__(InterceptionServer)
    interception.state_sessions = {"private": session}
    interception.state_service_secrets = frozenset({"shared"})
    interception.state_routes = {"route": session}
    app = web.Application()
    app.router.add_get("/state", interception.handle_state_get)

    lost_write_ack = False

    async def put(request):
        nonlocal lost_write_ack
        if (await request.json()).get("value") == 99:
            return web.json_response({"error": "fixture state rejection"}, status=400)
        response = await interception.handle_state_put(request)
        if not lost_write_ack and response.status == 200:
            lost_write_ack = True
            return web.json_response(
                {"error": "fixture lost PUT acknowledgement"}, status=500
            )
        return response

    lost_ack = False

    async def execution(request):
        nonlocal lost_ack
        response = await interception.handle_tool_execution(request)
        if (
            not lost_ack
            and response.status == 200
            and trace.tool_execution_events[
                json.loads(response.body)["receipt_seq"]
            ].phase
            == "returned"
        ):
            lost_ack = True
            return web.json_response(
                {"error": "fixture lost acknowledgement"}, status=500
            )
        return response

    app.router.add_put("/state", put)
    app.router.add_post("/tool-execution", execution)
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(TestServer(app))
        state_url = str(server.make_url("/state"))
        monkeypatch.setenv("VF_STATE_URL", state_url)
        monkeypatch.setenv("VF_STATE_SECRET", "private")
        tools = ReceiptTools(vf.ToolsetConfig())
        tools.arrived = 0
        tools.barrier = asyncio.Event()
        tools.entered = asyncio.Event()
        tools.hold = asyncio.Event()
        mcp = MCPServer("receipt-fixture")
        tools.register(mcp)
        mcp_app = mcp.streamable_http_app(json_response=True, stateless_http=True)
        await stack.enter_async_context(mcp.session_manager.run())
        mcp_client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mcp_app),
                base_url="http://localhost:8000",
            )
        )
        http_client = await stack.enter_async_context(httpx.AsyncClient())

        async def call(value, **kwargs):
            response = await mcp_client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": value,
                    "method": "tools/call",
                    "params": {
                        "name": "change",
                        "arguments": {"value": value, **kwargs},
                    },
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-06-18",
                },
            )
            response.raise_for_status()
            return response.json()

        failure = await call(4, fail=True)
        assert failure["result"]["isError"]
        assert trace.state.value == 0
        assert [event.phase for event in trace.tool_execution_events] == [
            "dispatch",
            "raised",
        ]
        failed = json.loads(trace.tool_execution_events[-1].receipt_json)
        assert failed["state_persistence"] == "not_attempted"
        assert json.loads(failed["evidence_json"][0])["local_after"] == 4
        results = await asyncio.wait_for(
            asyncio.gather(call(5, parallel=True), call(6, parallel=True)), timeout=5
        )
        assert all(not result["result"]["isError"] for result in results)
        returned = [
            json.loads(event.receipt_json)
            for event in trace.tool_execution_events
            if event.phase == "returned"
        ]
        assert sorted(item["state_write_revision"] for item in returned) == [1, 2]
        assert sorted(item["state_conflict"] for item in returned) == [False, True]
        assert trace.state.value in (5, 6)  # Existing last-write-wins behavior remains.
        assert trace.tool_state_revision == 2
        assert len(trace.state_write_receipts) == 2
        assert [event.receipt_seq for event in trace.tool_execution_events] == list(
            range(6)
        )
        original = trace.tool_execution_events[-1]
        payload = json.loads(original.receipt_json)
        endpoint = str(server.make_url("/tool-execution"))
        shared = {"Authorization": "Bearer shared", "X-Verifiers-State-Route": "route"}
        replay = await http_client.post(endpoint, json=payload, headers=shared)
        assert (
            replay.status_code == 200
            and replay.json()["receipt_seq"] == original.receipt_seq
        )
        assert len(trace.tool_execution_events) == 6
        payload["result_json"] = '"forged"'
        assert (
            await http_client.post(endpoint, json=payload, headers=shared)
        ).status_code == 400
        assert (await http_client.post(endpoint, json=payload)).status_code == 401
        # Applied claims require this occurrence's native write, not another
        # invocation's acknowledgement or merely a plausible host revision.
        valid_applied = next(
            item for item in returned if item["state_write_revision"] == 1
        )
        missing_ack = dict(valid_applied, invocation_id="no-native-write")
        assert (
            await http_client.post(endpoint, json=missing_ack, headers=shared)
        ).status_code == 400
        cross_claim = dict(valid_applied, state_write_revision=2, state_conflict=True)
        assert (
            await http_client.post(endpoint, json=cross_claim, headers=shared)
        ).status_code == 400
        wrong_read = dict(valid_applied, state_read_revision=1)
        assert (
            await http_client.post(endpoint, json=wrong_read, headers=shared)
        ).status_code == 400
        assert len(trace.tool_execution_events) == 6
        future = vf.ToolServerReceipt(
            invocation_id="future",
            event_index=0,
            phase="dispatch",
            tool_name="change",
            arguments_json="{}",
            state_read_revision=session.state_revision + 1,
        )
        assert (
            await http_client.post(
                endpoint, json=future.model_dump(mode="json"), headers=shared
            )
        ).status_code == 400
        write = trace.state_write_receipts[0]
        dispatch = next(
            event
            for event in trace.tool_execution_events
            if event.source == "tool_server"
            and event.invocation_id == write.write_id
            and event.phase == "dispatch"
        )
        written_value = json.loads(json.loads(dispatch.receipt_json)["arguments_json"])[
            "kwargs"
        ]["value"]
        write_headers = {
            "Authorization": "Bearer private",
            "X-Verifiers-State-Write-ID": write.write_id,
            "X-Verifiers-State-Expected-Revision": str(write.expected_revision),
        }
        before_replay = trace.state.model_dump()
        replayed_put = await http_client.put(
            state_url,
            json={"artifacts": {}, "value": written_value},
            headers=write_headers,
        )
        assert replayed_put.status_code == 200
        assert (
            int(replayed_put.headers["X-Verifiers-State-Revision"])
            == write.applied_revision
        )
        assert (
            trace.tool_state_revision == 2 and trace.state.model_dump() == before_replay
        )
        assert (
            await http_client.put(state_url, json={"value": 88}, headers=write_headers)
        ).status_code == 409
        future_write_headers = {
            "Authorization": "Bearer private",
            "X-Verifiers-State-Write-ID": "future-write",
            "X-Verifiers-State-Expected-Revision": "3",
        }
        assert (
            await http_client.put(
                state_url, json={"value": 88}, headers=future_write_headers
            )
        ).status_code == 400
        assert (
            trace.tool_state_revision == 2 and trace.state.model_dump() == before_replay
        )
        persisted_value = trace.state.value
        rejected = await call(99)
        assert rejected["result"]["isError"]
        assert trace.state.value == persisted_value
        persistence = json.loads(trace.tool_execution_events[-1].receipt_json)
        assert (
            persistence["phase"] == "returned"
            and json.loads(persistence["result_json"]) == "value:99"
        )
        assert (
            persistence["state_persistence"] == "failed"
            and persistence["state_error_json"] is not None
        )
        interrupted = asyncio.create_task(
            tools._with_state(tools.change)(value=7, hold=True)
        )
        await tools.entered.wait()
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
        assert trace.tool_execution_events[-1].phase == "interrupted"
        assert trace.state.value == persisted_value
        assert len(trace.tool_execution_events) == 10
        source = capture_trace_source(trace)
        restored = vf.WireTrace.model_validate(trace.to_record())
        assert restored.tool_execution_events == trace.tool_execution_events
        record = trace.to_record()
        no_ack_record = dict(record, state_write_receipts=[])
        with pytest.raises(ValueError, match="own native state-write acknowledgement"):
            vf.WireTrace.model_validate(no_ack_record)
        from verifiers.v1.assessments import canonical_json

        for field, value in (
            ("state_write_revision", 2),
            ("state_read_revision", 1),
            ("state_conflict", True),
        ):
            forged = json.loads(json.dumps(record))
            event = next(
                item
                for item in forged["tool_execution_events"]
                if item["source"] == "tool_server"
                and json.loads(item["receipt_json"]).get("state_write_revision") == 1
            )
            receipt = json.loads(event["receipt_json"])
            receipt[field] = value
            event["receipt_json"] = canonical_json(receipt)
            with pytest.raises(
                ValueError,
                match="disagrees with its native state-write acknowledgement",
            ):
                vf.WireTrace.model_validate(forged)
        forged_conflict = json.loads(json.dumps(record))
        forged_conflict["state_write_receipts"][0]["conflict"] = True
        with pytest.raises(
            ValueError, match="conflict disagrees with revision history"
        ):
            vf.WireTrace.model_validate(forged_conflict)
        resumed = RolloutSession(session.ctx, restored)
        assert resumed.state_revision == 2
        assert resumed.trace.state_write_receipts == trace.state_write_receipts
        retained = json.loads(source.source_json)
        assert retained["tool_state_revision"] == 2
        assert retained["state_write_receipts"] == [
            write.model_dump(mode="json") for write in trace.state_write_receipts
        ]
        trace_subject = vf.SubjectRef(
            kind="trace",
            snapshot_id=source.snapshot_id,
            episode_id="episode",
            trace_id=trace.id,
        )
        changed = restored.model_copy(deep=True)
        changed.tool_state_revision += 1
        assert (
            vf.project_subject(trace_subject, source, changed).reason
            == "current_tool_state_revision_changed"
        )
        changed = restored.model_copy(deep=True)
        changed.state_write_receipts = changed.state_write_receipts[:-1]
        assert (
            vf.project_subject(trace_subject, source, changed).reason
            == "current_state_write_receipts_changed"
        )
        assert (
            json.loads(source.source_json)["tool_execution_events"][-1]["source"]
            == "tool_server"
        )
        session.released = True
        assert (
            await http_client.post(
                endpoint, json=json.loads(original.receipt_json), headers=shared
            )
        ).status_code == 409
        assert (
            await http_client.put(
                state_url,
                json={"value": 98},
                headers={"Authorization": "Bearer private"},
            )
        ).status_code == 409
        assert trace.state.value in (5, 6)
