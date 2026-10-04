"""The per-rollout unit the interception layer serves.

One `RolloutSession` per rollout, registered on an interception server under the rollout's
secret. The rollout constructs it (model ctx, trace, task `@stop`s, limits) and the server
drives it: assigns its model client at register, routes each intercepted model call to it,
runs `refused()` before each turn, and stashes the real failure on `error`. `RolloutLimits` is the framework's per-rollout
budget (turns / tokens), checked between turns.
"""

import asyncio
import inspect
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import get_origin, get_type_hints
from uuid import uuid4

from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.assessments import canonical_json
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import RolloutError, TaskError
from verifiers.v1.trace import (
    InterceptRecord,
    ToolExecutionEvent,
    ToolServerExecutionEvent,
    Trace,
    validate_tool_server_state_ack,
)
from verifiers.v1.types import (
    AssistantMessage,
    Messages,
    Request,
    Response,
    ToolMessage,
    UserMessage,
    generated_arguments_equal,
)
from verifiers.v1.utils.decorators import invoke

logger = logging.getLogger(__name__)


def hook_boundary(handler: Callable, *, allow_trace: bool) -> type:
    """Select a hook boundary solely from its annotated parameters."""
    hints = get_type_hints(handler)
    annotations = [
        get_origin(hints[name]) or hints[name]
        for name in inspect.signature(handler).parameters
        if name in hints
    ]
    boundaries = [kind for kind in annotations if kind in (Request, Response)]
    if len(boundaries) == 1 and annotations.count(Trace) <= 1:
        return boundaries[0]
    if not boundaries and allow_trace and annotations.count(Trace) == 1:
        return Trace
    expected = "Request, Response, or Trace" if allow_trace else "Request or Response"
    raise TypeError(f"{handler.__name__} must have exactly one {expected} parameter")


async def call_hook(handler: Callable, available: dict[type, object]) -> object:
    result = invoke(handler, available)
    return await result if inspect.isawaitable(result) else result


@dataclass(frozen=True)
class RolloutLimits:
    """Per-rollout framework limits (None = no cap), checked before each turn is served.
    The first limit reached refuses the turn — halting any harness, the same mechanism as
    a @stop — and becomes the trace's stop condition. Each caps a trace computed property:
    `max_turns` -> num_turns, `max_input_tokens` -> num_input_tokens, `max_output_tokens` ->
    num_output_tokens, `max_total_tokens` -> num_total_tokens. Token caps are soft by one turn:
    they're checked between turns, so the turn that crosses a cap still completes."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def reached(self, trace: Trace) -> str | None:
        """The name of the first limit `trace` has reached, or None if within all caps."""
        if self.max_turns is not None and trace.num_turns >= self.max_turns:
            return "max_turns"
        if (
            self.max_input_tokens is not None
            and trace.num_input_tokens >= self.max_input_tokens
        ):
            return "max_input_tokens"
        if (
            self.max_output_tokens is not None
            and trace.num_output_tokens >= self.max_output_tokens
        ):
            return "max_output_tokens"
        if (
            self.max_total_tokens is not None
            and trace.num_total_tokens >= self.max_total_tokens
        ):
            return "max_total_tokens"
        return None


@dataclass
class IdempotentRequest:
    """One non-streaming model request shared by its original call and retries."""

    binding: tuple[str, bytes]
    response: "ReplayResponse | None" = None
    completed_at: float | None = None
    inflight: "asyncio.Future[ReplayResponse | None] | None" = None


@dataclass(frozen=True)
class ReplayResponse:
    """The exact HTTP result handed to coalesced in-flight request attempts."""

    status: int
    body: bytes


@dataclass
class RolloutSession:
    ctx: ModelContext
    trace: Trace
    network_policy: NetworkPolicyConfig = field(default_factory=NetworkPolicyConfig)
    """The resolved execution policy, including task-level restrictions."""
    trace_stops: list[Callable[..., Awaitable[bool] | bool]] = field(
        default_factory=list
    )
    limits: RolloutLimits = field(default_factory=RolloutLimits)
    request_interceptors: list[Callable] = field(default_factory=list)
    response_interceptors: list[Callable] = field(default_factory=list)
    request_stops: list[Callable] = field(default_factory=list)
    response_stops: list[Callable] = field(default_factory=list)
    client: Client | None = None
    """The model client serving this rollout's turns. The interception server assigns it at
    `register` (one server-owned client per distinct endpoint config), so every rollout it
    multiplexes shares one keepalive connection pool instead of opening its own."""
    error: "RolloutError | None" = None
    """The latest unresolved model-call failure. The harness only sees it as an HTTP error, so
    when its program dies on it the rollout records this original error instead of a secondary
    `HarnessError`. A harness that completes cleanly after the failure handled it. Reset before
    each model turn, so a successful retry clears it."""
    idempotent_requests: dict[str, IdempotentRequest] = field(default_factory=dict)
    """Explicit keys or marked SDK retries mapped to their replay state."""
    released: bool = False
    """Set when the rollout unregisters the session: the trace is sealed (its conclusion is
    what scored and persisted), so a handler still in flight must not commit turns, record
    calls, or write state onto it — the in-memory trace must stay what the run produced."""
    tasks: set["asyncio.Task"] = field(default_factory=set)
    """Handler tasks currently serving this session. aiohttp does not cancel a handler when
    its client disconnects, so a request whose program died at teardown would keep driving
    the exchange (upstream call, simulator turn) — unregistering cancels these instead."""
    prepared_tool_results: dict[str, ToolMessage] = field(default_factory=dict)
    prepared_users: Counter[str] = field(default_factory=Counter)
    tool_receipt_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @property
    def state_revision(self) -> int:
        return self.trace.tool_state_revision

    @state_revision.setter
    def state_revision(self, value: int) -> None:
        self.trace.tool_state_revision = value

    def retain_tool_server_receipt(self, receipt):
        """Synchronously acknowledge one immutable server report; never assign action credit."""
        from verifiers.v1.mcp.execution import ToolServerReceipt

        receipt = ToolServerReceipt.model_validate(
            receipt.model_dump(mode="python"), strict=True
        )
        if self.released:
            raise TaskError("tool-server receipt arrived after rollout was sealed")
        from verifiers.v1._execution_links import validate_server_parent

        validate_server_parent(
            receipt, self.trace.tool_execution_events, self.trace.nodes
        )
        validate_tool_server_state_ack(receipt, self.trace.state_write_receipts)
        if any(
            value is not None and value > self.state_revision
            for value in (receipt.state_read_revision, receipt.state_write_revision)
        ):
            raise TaskError("tool-server receipt reports a future state revision")
        payload = canonical_json(receipt.model_dump(mode="json"))
        previous = [
            item
            for item in self.trace.tool_execution_events
            if isinstance(item, ToolServerExecutionEvent)
            and item.invocation_id == receipt.invocation_id
        ]
        for item in previous:
            if item.event_index == receipt.event_index:
                if item.receipt_json != payload:
                    raise TaskError("conflicting duplicate tool-server receipt")
                return {
                    "ok": True,
                    "receipt_seq": item.receipt_seq,
                    "state_revision": item.state_revision,
                }
        if (not previous and receipt.phase != "dispatch") or (
            previous and (len(previous) != 1 or receipt.event_index != 1)
        ):
            raise TaskError("invalid tool-server execution lifecycle")
        if previous:
            original = json.loads(previous[0].receipt_json)
            current = receipt.model_dump(mode="json")
            if any(
                original.get(key) != current.get(key)
                for key in (
                    "tool_name",
                    "arguments_json",
                    "state_read_revision",
                    "parent_execution_id",
                    "dispatch_ticket",
                    "transport_attempt_index",
                    "server_name",
                )
            ):
                raise TaskError("tool-server execution changed invocation identity")
        item = ToolServerExecutionEvent(
            invocation_id=receipt.invocation_id,
            event_index=receipt.event_index,
            phase=receipt.phase,
            receipt_seq=len(self.trace.tool_execution_events),
            state_revision=self.state_revision,
            receipt_json=payload,
        )
        self.trace.tool_execution_events += (item,)
        return {
            "ok": True,
            "receipt_seq": item.receipt_seq,
            "state_revision": item.state_revision,
        }

    @property
    def stopped(self) -> bool:
        return self.trace.stop_condition is not None

    async def rewrite_request(
        self, request: Request, *, run_stops: bool = True
    ) -> tuple[Request, list[InterceptRecord], str | None]:
        """Run typed request interceptors and stops over one canonical request."""
        if not self.request_interceptors and (not run_stops or not self.request_stops):
            return request, [], None
        prepared: set[int] = set()
        candidates: set[int] = set()
        if self.request_interceptors:
            turn = graph.prepare_turn(self.trace, request.messages)
            prepared_users = self.prepared_users.copy()
            for position in range(turn.tail_start, len(request.messages)):
                message = request.messages[position]
                if isinstance(message, UserMessage):
                    candidates.add(position)
                    if prepared_users:
                        key = graph.message_hash(message)
                        if prepared_users[key]:
                            prepared_users[key] -= 1
                            prepared.add(position)
                elif isinstance(message, ToolMessage):
                    candidates.add(position)
                    if self.prepared_tool_results.get(message.tool_call_id) == message:
                        prepared.add(position)
        already_intercepted = candidates and candidates == prepared

        current = request
        records: list[InterceptRecord] = []
        try:
            interceptors = [] if already_intercepted else self.request_interceptors
            for handler in interceptors:
                candidate = current.model_copy(deep=True)
                result = await call_hook(
                    handler, {Request: candidate, Trace: self.trace}
                )
                if result is None:
                    continue
                if not isinstance(result, Request):
                    raise TypeError(f"expected Request, got {type(result).__name__}")
                if len(result.messages) != len(current.messages):
                    raise ValueError(
                        "request interceptors cannot add or remove messages"
                    )
                if result.tools != current.tools:
                    raise ValueError("request interceptors cannot rewrite tools")
                for position, (before, after) in enumerate(
                    zip(current.messages, result.messages, strict=True)
                ):
                    if before == after:
                        continue
                    if position not in candidates - prepared:
                        raise ValueError(
                            "request interceptors can only rewrite new user or tool messages"
                        )
                    if type(after) is not type(before):
                        raise TypeError(
                            f"expected {type(before).__name__}, got {type(after).__name__}"
                        )
                    if (
                        isinstance(before, ToolMessage)
                        and after.tool_call_id != before.tool_call_id
                    ):
                        raise ValueError(
                            "request interceptors cannot change a tool-call ID"
                        )
                if result != current:
                    current = result
                    records.append(InterceptRecord(handler=handler.__name__))

            stops = self.request_stops if run_stops else []
            for stop in stops:
                candidate = current.model_copy(deep=True)
                result = await call_hook(stop, {Request: candidate, Trace: self.trace})
                if not isinstance(result, bool):
                    raise TypeError(
                        f"@stop must return bool, got {type(result).__name__}"
                    )
                if result:
                    return current, records, stop.__name__
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"request interception failed: {type(error).__name__}: {error}"
            ) from error
        return current, records, None

    def consume_prepared(self, messages: Messages) -> None:
        """Forget pre-harness rewrites only after their model request commits."""
        for message in messages:
            if isinstance(message, UserMessage) and self.prepared_users:
                key = graph.message_hash(message)
                if self.prepared_users[key]:
                    self.prepared_users[key] -= 1
            elif isinstance(message, ToolMessage):
                self.prepared_tool_results.pop(message.tool_call_id, None)

    async def prepare_users(
        self, request: Request
    ) -> tuple[Request, list[InterceptRecord]]:
        """Intercept caller-owned user turns before the harness stores them."""
        branch = self.trace.messages
        rewritten, records, _ = await self.rewrite_request(
            Request(messages=[*branch, *request.messages]), run_stops=False
        )
        tail = rewritten.messages[len(branch) :]
        self.prepared_users.update(
            graph.message_hash(message)
            for message in tail
            if isinstance(message, UserMessage)
        )
        return Request(messages=tail), records

    async def rewrite_response(
        self, response: Response
    ) -> tuple[Response, list[InterceptRecord], str | None]:
        """Run typed response interceptors and stops before harness delivery."""
        records: list[InterceptRecord] = []
        try:
            for handler in self.response_interceptors:
                candidate = response.model_copy(deep=True)
                result = await call_hook(
                    handler, {Response: candidate, Trace: self.trace}
                )
                if result is None:
                    continue
                if not isinstance(result, Response):
                    raise TypeError(f"expected Response, got {type(result).__name__}")
                if result == response:
                    continue
                unchanged = result.model_copy(
                    update={
                        "message": response.message,
                        "finish_reason": response.finish_reason,
                    }
                )
                if unchanged != response:
                    raise ValueError(
                        "response interceptors can only replace the assistant message"
                    )
                if (
                    result.message.reasoning_content
                    or result.message.tool_calls
                    or result.message.provider_state
                ):
                    raise ValueError(
                        "response interceptors must return an inert text-only message"
                    )
                response = result.model_copy(update={"finish_reason": "stop"})
                records.append(InterceptRecord(handler=handler.__name__))

            for stop in self.response_stops:
                candidate = response.model_copy(deep=True)
                result = await call_hook(stop, {Response: candidate, Trace: self.trace})
                if not isinstance(result, bool):
                    raise TypeError(
                        f"@stop must return bool, got {type(result).__name__}"
                    )
                if result:
                    return response, records, stop.__name__
        except RolloutError:
            raise
        except Exception as error:
            raise TaskError(
                f"response interception failed: {type(error).__name__}: {error}"
            ) from error
        return response, records, None

    async def handle_tool(
        self, phase: str, message: ToolMessage, *, request=None
    ) -> dict:
        """Retain occurrence-bound receipts; duplicate deliveries never replay hooks."""
        from verifiers.v1.interception.tool import ToolHookRequest

        hook = ToolHookRequest.model_validate(
            (request or ToolHookRequest(phase=phase, message=message)).model_dump(
                mode="python"
            ),
            strict=True,
        )
        if hook.phase != phase or hook.message != message:
            raise TaskError("tool receipt arguments disagree with payload")
        if self.released:
            raise TaskError("tool receipt arrived after rollout was sealed")
        if hook.execution_id is None:
            return await self._handle_tool_policy(phase, message)
        lock = self.tool_receipt_locks.setdefault(hook.execution_id, asyncio.Lock())
        async with lock:
            if self.released:
                raise TaskError("tool receipt arrived after rollout was sealed")
            payload = canonical_json(hook.model_dump(mode="json"))
            previous = [
                event
                for event in self.trace.tool_execution_events
                if isinstance(event, ToolExecutionEvent)
                and event.execution_id == hook.execution_id
            ]
            for event in previous:
                if event.event_index == hook.event_index:
                    if event.request_json != payload:
                        raise TaskError("conflicting duplicate tool execution receipt")
                    return json.loads(event.decision_json)
            frontier = {
                id(branch.nodes[-1]) for branch in self.trace.branches if branch.nodes
            }
            matches = [
                (index, ordinal, node)
                for index, node in enumerate(self.trace.nodes)
                if isinstance(node.message, AssistantMessage)
                and (
                    index == previous[0].node_index
                    if previous
                    else id(node) in frontier
                )
                for ordinal, call in enumerate(node.message.tool_calls or [])
                if call.id == hook.call.id
                and (ordinal == previous[0].emitted_call_index if previous else True)
            ]
            if len(matches) != 1:
                raise TaskError(
                    "tool execution must match one unique native emitted call"
                )
            node_index, ordinal, node = matches[0]
            emitted = node.message.tool_calls[ordinal]
            if (
                emitted.name != hook.call.name
                or emitted.type != hook.call.type
                or not generated_arguments_equal(emitted.arguments, hook.call.arguments)
            ):
                raise TaskError(
                    "tool execution payload disagrees with native emitted call"
                )
            expected = 0 if not previous else previous[-1].event_index + 1
            if hook.event_index != expected:
                raise TaskError(
                    "tool execution receipts must have contiguous event indices"
                )
            if not previous:
                allowed = hook.phase == "before"
            elif previous[-1].phase == "before":
                allowed = hook.phase in ("dispatch", "rejected")
                if json.loads(previous[-1].decision_json).get("action") != "allow":
                    allowed = False
            elif previous[-1].phase == "dispatch":
                allowed = hook.phase in ("after", "raised", "interrupted")
            else:
                allowed = False
            if not allowed:
                raise TaskError("invalid tool execution lifecycle transition")
            if previous and (
                previous[0].node_index,
                previous[0].emitted_call_index,
            ) != (node_index, ordinal):
                raise TaskError("tool execution occurrence changed its native target")
            attempts = [
                attempt.attempt_index
                for attempt in node.generated_calls
                if attempt.emitted_call_index == ordinal and node.sampled
            ]
            generated_index = (
                previous[0].generated_attempt_index
                if previous
                else attempts[0]
                if len(attempts) == 1
                else None
            )
            if (
                previous
                and generated_index is not None
                and attempts != [generated_index]
            ):
                raise TaskError(
                    "tool execution generated attribution changed during occurrence"
                )

            def retain(decision):
                if self.released:
                    return
                event = ToolExecutionEvent(
                    execution_id=hook.execution_id,
                    event_index=hook.event_index,
                    phase=hook.phase,
                    receipt_seq=len(self.trace.tool_execution_events),
                    node_index=node_index,
                    emitted_call_index=ordinal,
                    generated_attempt_index=generated_index,
                    request_json=payload,
                    decision_json=canonical_json(decision),
                )
                self.trace.tool_execution_events += (event,)

            try:
                decision = (
                    await self._handle_tool_policy(
                        "after" if phase == "rejected" else phase, message
                    )
                    if phase in ("before", "after", "rejected")
                    else {"action": "allow"}
                )
            except asyncio.CancelledError:
                retain({"action": "incomplete", "reason": "interceptor_cancelled"})
                raise
            except Exception as error:
                retain({"action": "error", "error": f"{type(error).__name__}: {error}"})
                raise
            if self.released:
                raise TaskError("tool receipt finished after rollout was sealed")
            if hook.mcp_dispatch is not None:
                from verifiers.v1._execution_links import validate_dispatch_call

                validate_dispatch_call(hook)
                if node.sampled is not True:
                    raise TaskError("MCP dispatch must target an original sampled call")
                decision = {**decision, "mcp_dispatch_ticket": uuid4().hex}
            retain(decision)
            return decision

    async def _handle_tool_policy(self, phase: str, message: ToolMessage) -> dict:
        """Intercept a harness-owned tool result before the harness records it."""
        branches = [
            branch
            for branch in self.trace.branches
            if branch.nodes
            and isinstance(branch.nodes[-1].message, AssistantMessage)
            and any(
                call.id == message.tool_call_id
                for call in branch.nodes[-1].message.tool_calls or []
            )
        ]
        if len(branches) != 1:
            raise TaskError(
                f"tool call {message.tool_call_id!r} matched {len(branches)} branches"
            )
        branch = branches[0]
        assistant = branch.nodes[-1].message
        assert isinstance(assistant, AssistantMessage)
        # Keep earlier results in the hook's trace, but commit them only when the model
        # request arrives and can supply their token attribution.
        previous = [
            self.prepared_tool_results[call.id]
            for call in assistant.tool_calls or []
            if call.id in self.prepared_tool_results
        ]
        request, records, stopped = await self.rewrite_request(
            Request(
                messages=[*branch.messages, *previous, message],
                tools=self.trace.tools or None,
            )
        )
        if self.released:
            raise TaskError("tool policy finished after rollout was sealed")
        candidate = request.messages[-1]
        assert isinstance(candidate, ToolMessage)
        self.trace.request_rewrites.extend(records)
        if stopped is not None:
            committed = request.messages if phase == "after" else request.messages[:-1]
            turn = graph.prepare_turn(self.trace, committed)
            turn.commit_prompt()
            self.consume_prepared(turn.tail)
            self.trace.stop(stopped)
            return {"action": "stop", "reason": stopped}
        if phase == "before" and candidate == message:
            return {"action": "allow"}
        self.prepared_tool_results[candidate.tool_call_id] = candidate
        if candidate != message:
            return {
                "action": "rewrite",
                "message": candidate.model_dump(exclude_none=True),
            }
        return {"action": "allow"}

    @cached_property
    def state_adapter(self) -> TypeAdapter:
        """The rollout's state codec, built only when a state channel is used."""
        return TypeAdapter(type(self.trace.state))

    def adopt(self, task: "asyncio.Task | None") -> None:
        """Track a handler task serving this session, for cancellation at release.
        Callers adopt in the same synchronous stretch that fetched the session, so
        `release()` can't interleave; the released check keeps the seal even if a
        future caller breaks that invariant (an await before adopting)."""
        if task is None:
            return
        if self.released:  # sealed while this handler was scheduled — don't serve
            task.cancel()
            return
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def release(self) -> None:
        """Seal the session: no further trace mutation, and in-flight handlers cancel."""
        self.released = True
        for task in list(self.tasks):
            task.cancel()

    async def refused(self) -> str | None:
        """The framework's limits (turns / token budget) and `@stop` checks, run before each
        model call. Sets the stop condition and returns its name, else None. A refused first
        call halts the harness (its model call errors out); HarnessSession.turn treats it as clean. A task
        that ends a trajectory from `trace.state` does it with its own `@stop` (run here generically),
        so the interception server holds no opinion about the state's contents."""
        if (limit := self.limits.reached(self.trace)) is not None:
            self.trace.stop(limit)
            logger.debug("limit %r reached: id=%s", limit, self.trace.id)
            return limit
        for stop in self.trace_stops:
            result = await call_hook(stop, {Trace: self.trace})
            if not isinstance(result, bool):
                raise TaskError(f"@stop must return bool, got {type(result).__name__}")
            if result:
                self.trace.stop(stop.__name__)
                logger.debug("stop %r fired: id=%s", stop.__name__, self.trace.id)
                return stop.__name__
        return None
