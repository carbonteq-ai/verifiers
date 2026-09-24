"""Per-request context budget: `prompt + max_tokens` stays inside `max_total_tokens`.

Reproduces a 64k-context evaluation where the previous call ended at
55,963 prompt + 526 completion tokens, the next prompt measured 57,345 tokens and
the fixed `max_tokens: 8192` made the provider reject the call (57,345 + 8,192 =
65,537 > 65,536). The between-turn `max_total_tokens` check could not see it: it
reads the previous call's 56,489-token sequence."""

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients import ModelContext
from verifiers.v1.clients.client import Client
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import (
    CONTEXT_MESSAGE_OVERHEAD_TOKENS,
    CONTEXT_SAFETY_MARGIN_TOKENS,
    MIN_CONTEXT_OUTPUT_TOKENS,
    RolloutLimits,
    RolloutSession,
    message_token_upper_bound,
    prompt_token_upper_bound,
)
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import ToolCall, Usage

CONTEXT = 65_536
MAX_TOKENS = 8_192

SYSTEM = vf.SystemMessage(content="You operate business apps through tools.")
USER = vf.UserMessage(content="Reconcile the open invoices.")
CALL = ToolCall(id="call_19", name="bash", arguments='{"cmd": "list_invoices"}')
ASSISTANT = vf.AssistantMessage(content=None, tool_calls=[CALL])
# ~850 tokens of tool output, as in the observed failing turn.
TOOL = vf.ToolMessage(tool_call_id="call_19", content='{"invoice": "INV-1"}\n' * 150)


def _trace(prompt_tokens: int, completion_tokens: int) -> vf.Trace:
    """A trace whose last committed turn reported the given usage."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    node = graph.prepare_turn(trace, [SYSTEM, USER]).commit(
        vf.Response(
            id="r",
            created=0,
            model="m",
            message=ASSISTANT,
            finish_reason="tool_calls",
        )
    )
    trace.calls.append(
        ModelCall(
            node=node,
            finish_reason="tool_calls",
            usage=Usage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )
    )
    return trace


def test_tool_message_bound_counts_utf8_bytes_plus_overhead():
    message = vf.ToolMessage(tool_call_id="c", content="héllo")
    assert message_token_upper_bound(message) == CONTEXT_MESSAGE_OVERHEAD_TOKENS + 7
    image = vf.UserMessage(
        content=[vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url="data:x"))]
    )
    assert message_token_upper_bound(image) is None


def test_prompt_bound_anchors_on_previous_call_usage():
    trace = _trace(55_963, 526)
    turn = graph.prepare_turn(trace, [SYSTEM, USER, ASSISTANT, TOOL])
    assert turn.tail == [TOOL]
    bound = prompt_token_upper_bound(turn)
    tool_bound = message_token_upper_bound(TOOL)
    assert tool_bound is not None
    assert bound == 55_963 + 526 + tool_bound + CONTEXT_MESSAGE_OVERHEAD_TOKENS
    # An upper bound on the provider's measured 57,345-token prompt.
    assert bound >= 57_345


def test_first_turn_is_not_estimated():
    trace = _trace(55_963, 526)
    fresh = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    assert prompt_token_upper_bound(graph.prepare_turn(fresh, [SYSTEM, USER])) is None
    # A prompt that no longer matches the recorded history has no anchor either.
    rewritten = vf.UserMessage(content="summarized history")
    assert (
        prompt_token_upper_bound(graph.prepare_turn(trace, [SYSTEM, rewritten])) is None
    )


def test_output_room_reproduces_observed_overflow():
    trace = _trace(55_963, 526)
    turn = graph.prepare_turn(trace, [SYSTEM, USER, ASSISTANT, TOOL])
    limits = RolloutLimits(max_total_tokens=CONTEXT)
    # The between-turn check cannot see the overflow ...
    assert limits.reached(trace) is None
    room = limits.output_room(turn)
    assert room is not None
    # ... but the per-request room does: 57,345 + 8,192 would be 65,537 > 65,536.
    assert MIN_CONTEXT_OUTPUT_TOKENS <= room < MAX_TOKENS
    assert 57_345 + room <= CONTEXT - CONTEXT_SAFETY_MARGIN_TOKENS
    assert RolloutLimits().output_room(turn) is None


def test_cap_max_tokens_rewrites_present_keys_only():
    chat = ChatDialect()
    assert chat.cap_max_tokens({"max_tokens": 8192}, 5000) == {"max_tokens": 5000}
    assert chat.cap_max_tokens({"max_completion_tokens": 8192}, 5000) == {
        "max_completion_tokens": 5000
    }
    assert chat.cap_max_tokens({"max_tokens": 4000}, 5000) is None
    # No cap requested: the provider's own default (remaining context on vLLM) applies.
    assert chat.cap_max_tokens({"temperature": 1.0}, 5000) is None
    assert ResponsesDialect().cap_max_tokens({"max_output_tokens": 8192}, 5000) == {
        "max_output_tokens": 5000
    }


class _RecordingClient(Client):
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def get_response(
        self, dialect, body, sampling, session_id=None, turn=None, headers=None
    ):
        self.bodies.append(body)
        message = vf.AssistantMessage(content="done")
        response = vf.Response(
            id="r2",
            created=0,
            model=body["model"],
            message=message,
            finish_reason="stop",
            usage=Usage(prompt_tokens=57_345, completion_tokens=3),
        )
        response.raw = {
            "id": "r2",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 57_345,
                "completion_tokens": 3,
                "total_tokens": 57_348,
            },
        }
        return response


def _wire(message: vf.Message) -> dict:
    if isinstance(message, vf.AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls or []
            ],
        }
    return message.model_dump(exclude_none=True)


async def _post(trace: vf.Trace) -> tuple[httpx.Response, _RecordingClient]:
    client = _RecordingClient()
    session = RolloutSession(
        ctx=ModelContext(
            "policy",
            EvalClientConfig(base_url="http://127.0.0.1:1/v1"),
            vf.Sampling(max_tokens=MAX_TOKENS),
        ),
        trace=trace,
        limits=RolloutLimits(max_total_tokens=CONTEXT),
    )
    async with (
        InterceptionServer(client_factory=lambda config: client) as server,
        server.acquire(session) as (base_url, secret, _),
        httpx.AsyncClient() as http,
    ):
        response = await http.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
            json={
                "model": "anything",
                "max_tokens": 32_000,
                "messages": [_wire(m) for m in (SYSTEM, USER, ASSISTANT, TOOL)],
            },
        )
    return response, client


@pytest.mark.asyncio
async def test_server_clamps_request_that_would_overflow_context():
    trace = _trace(55_963, 526)
    response, client = await _post(trace)
    assert response.status_code == 200, response.text
    [body] = client.bodies
    assert MIN_CONTEXT_OUTPUT_TOKENS <= body["max_tokens"] < MAX_TOKENS
    assert 57_345 + body["max_tokens"] <= CONTEXT
    # The clamped value is what the trace records as this call's sampling.
    assert trace.calls[-1].sampling.max_tokens == body["max_tokens"]
    assert trace.stop_condition is None


@pytest.mark.asyncio
async def test_server_stops_as_truncation_when_context_is_exhausted():
    trace = _trace(64_800, 400)
    response, client = await _post(trace)
    assert response.status_code == 400
    assert "rollout stopped: max_total_tokens" in response.text
    assert client.bodies == []  # nothing the provider would reject was sent
    assert trace.stop_condition == "max_total_tokens"
    assert trace.is_truncated
    # No call error is recorded: a consumer counts this as truncation, not a failure.
    assert all(call.error is None for call in trace.calls)
    # The final tool result is kept in the graph for inspection.
    last = trace.nodes[-1].message
    assert isinstance(last, vf.ToolMessage) and last.content == TOOL.content
