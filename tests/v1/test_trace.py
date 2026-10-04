"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import asyncio
import json
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.semantic import (
    ACP_EXTENSION_HEADERS,
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    extract_acp_info,
)
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


@pytest.mark.asyncio
async def test_tool_execution_receipts_preserve_raw_result_and_replay_decisions():
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.errors import TaskError
    from verifiers.v1.interception.tool import ToolHookRequest
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.types import ToolCall, ToolMessage

    call = ToolCall(id="call", name="send", arguments='{"a":1}')
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="send")),
        nodes=[
            MessageNode(
                message=AssistantMessage(content="", tool_calls=[call]), sampled=True
            )
        ],
    )
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    message = ToolMessage(tool_call_id=call.id, content="before")
    before = ToolHookRequest(
        phase="before",
        message=message,
        execution_id="execution",
        call=call,
        event_index=0,
    )
    assert await session.handle_tool("before", message, request=before) == {
        "action": "allow"
    }
    assert await session.handle_tool("before", message, request=before) == {
        "action": "allow"
    }
    assert len(trace.tool_execution_events) == 1
    conflicting = before.model_copy(
        update={"message": message.model_copy(update={"content": "changed"})}
    )
    with pytest.raises(TaskError, match="conflicting duplicate"):
        await session.handle_tool("before", conflicting.message, request=conflicting)
    dispatch = before.model_copy(update={"phase": "dispatch", "event_index": 1})
    await session.handle_tool("dispatch", message, request=dispatch)
    after = before.model_copy(
        update={
            "phase": "after",
            "event_index": 2,
            "message": message.model_copy(update={"content": "bounded"}),
            "raw_result": "raw" * 10000,
        }
    )
    await session.handle_tool("after", after.message, request=after)
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert restored.tool_execution_events == trace.tool_execution_events
    payload = json.loads(restored.tool_execution_events[-1].request_json)
    assert payload["raw_result"] == "raw" * 10000
    assert payload["message"]["content"] == "bounded"
    source = capture_trace_source(trace)
    assert json.loads(source.source_json)["tool_execution_events"] == [
        event.model_dump(mode="json") for event in trace.tool_execution_events
    ]
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id=trace.id,
    )
    assert vf.project_subject(subject, source, restored).status == "unsupported"
    restored.tool_execution_events = restored.tool_execution_events[:-1]
    assert (
        vf.project_subject(subject, source, restored).reason
        == "current_tool_execution_events_changed"
    )
    # Fallback provider IDs may recur on a later turn; occurrence binding is native.
    trace.nodes.append(
        MessageNode(
            parent=0,
            message=AssistantMessage(content="new", tool_calls=[call]),
            sampled=True,
        )
    )
    newer = before.model_copy(update={"execution_id": "later-execution"})
    await session.handle_tool("before", message, request=newer)
    assert trace.tool_execution_events[-1].node_index == 1
    duplicate = vf.Trace(
        episode_id="episode",
        agent=trace.agent,
        task=trace.task,
        nodes=[
            MessageNode(
                message=AssistantMessage(content="", tool_calls=[call, call]),
                sampled=True,
            )
        ],
    )
    duplicate_session = RolloutSession(session.ctx, duplicate)
    with pytest.raises(TaskError, match="unique native"):
        await duplicate_session.handle_tool("before", message, request=before)
    assert duplicate.tool_execution_events == ()
    for changes in ({"error": "invalid"}, {"raw_result": "invalid"}):
        with pytest.raises(ValueError):
            ToolHookRequest.model_validate({**before.model_dump(), **changes})
    with pytest.raises(ValueError):
        ToolHookRequest.model_validate(
            {**before.model_dump(), "phase": "raised", "raw_result": "invalid"}
        )
    # Replay may not strengthen an originally absent generated attribution.
    forged = trace.to_record()
    forged["nodes"][0]["generated_calls"] = [
        vf.GeneratedCallAttempt(
            attempt_index=0,
            emitted_call_index=0,
            provider_call_id="call",
            parse_status="parsed",
            raw="send",
            name="send",
            arguments='{"a":1}',
            coordinate_system="node_local_full_tokens",
            parser_revision="fixture",
            span_fidelity="unavailable",
            completion_token_digest="fixture",
        ).model_dump(mode="json")
    ]
    forged["tool_execution_events"][1]["generated_attempt_index"] = 0
    with pytest.raises(ValueError, match="lifecycle"):
        vf.WireTrace.model_validate(forged)
    unsampled = vf.Trace(
        episode_id="episode",
        agent=trace.agent,
        task=trace.task,
        nodes=[
            MessageNode(
                message=AssistantMessage(content="", tool_calls=[call]),
                generated_calls=tuple(
                    vf.GeneratedCallAttempt.model_validate(item)
                    for item in forged["nodes"][0]["generated_calls"]
                ),
            )
        ],
    )
    external = RolloutSession(session.ctx, unsampled)
    await external.handle_tool("before", message, request=before)
    assert unsampled.tool_execution_events[0].generated_attempt_index is None
    forged_unsampled = unsampled.to_record()
    forged_unsampled["tool_execution_events"][0]["generated_attempt_index"] = 0
    with pytest.raises(ValueError, match="unsampled"):
        vf.WireTrace.model_validate(forged_unsampled)
    # Separate occurrences can enter awaited hooks concurrently; same-occurrence
    # duplicates wait for one policy decision rather than replaying it.
    calls = [call, call.model_copy(update={"id": "second"})]
    concurrent = vf.Trace(
        episode_id="episode",
        agent=trace.agent,
        task=trace.task,
        nodes=[
            MessageNode(
                message=AssistantMessage(content="", tool_calls=calls), sampled=True
            )
        ],
    )
    arrived = 0
    barrier = asyncio.Event()

    async def policy(request: vf.Request):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            barrier.set()
        await barrier.wait()

    active = RolloutSession(session.ctx, concurrent, request_interceptors=[policy])
    requests = [
        ToolHookRequest(
            phase="before",
            message=ToolMessage(tool_call_id=item.id, content="before"),
            execution_id=item.id,
            call=item,
            event_index=0,
        )
        for item in calls
    ]
    await asyncio.wait_for(
        asyncio.gather(
            *(
                active.handle_tool("before", item.message, request=item)
                for item in [requests[0], requests[1], requests[0]]
            )
        ),
        timeout=2,
    )
    assert arrived == 2
    assert [event.receipt_seq for event in concurrent.tool_execution_events] == [0, 1]

    async def failing_policy(request: vf.Request):
        raise ValueError("policy failed")

    failed = RolloutSession(
        session.ctx,
        concurrent.model_copy(deep=True),
        request_interceptors=[failing_policy],
    )
    failed.trace.tool_execution_events = ()
    with pytest.raises(TaskError, match="policy failed"):
        await failed.handle_tool("before", requests[0].message, request=requests[0])
    assert (
        json.loads(failed.trace.tool_execution_events[0].decision_json)["action"]
        == "error"
    )
    assert (
        await failed.handle_tool("before", requests[0].message, request=requests[0])
    )["action"] == "error"
    for phase in ("dispatch", "rejected"):
        terminal = requests[0].model_copy(update={"phase": phase, "event_index": 1})
        with pytest.raises(TaskError, match="lifecycle"):
            await failed.handle_tool(phase, terminal.message, request=terminal)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def delayed_policy(request: vf.Request):
        entered.set()
        await finish.wait()
        rewritten = request.model_copy(deep=True)
        rewritten.messages[-1].content = "late rewrite"
        return rewritten

    sealed = RolloutSession(
        session.ctx,
        concurrent.model_copy(deep=True),
        request_interceptors=[delayed_policy],
    )
    sealed.trace.tool_execution_events = ()
    initial = sealed.trace.to_record()
    waiting = asyncio.create_task(
        sealed.handle_tool("before", requests[0].message, request=requests[0])
    )
    await entered.wait()
    sealed.released = True
    finish.set()
    with pytest.raises(TaskError, match="sealed"):
        await waiting
    assert sealed.trace.to_record() == initial
    assert sealed.prepared_tool_results == {}


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )
    trace.root_reply = "previous reply"

    class FailingSession:
        async def turn(self, messages):
            trace.nodes.append(
                MessageNode(
                    parent=None,
                    message=AssistantMessage(content="current partial reply"),
                    sampled=True,
                )
            )
            raise RuntimeError("segment failed after sampling")

    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = None
    run._timeouts = RolloutTimeouts()
    run._harness_session = FailingSession()
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
    )
    run.deadline_at = None

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


def test_native_assessments_survive_trace_and_episode_records():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="hello")),
    )
    source = vf.SourceSnapshot.capture(
        {"trace_id": trace.id, "prompt": "hello"},
        episode_id="episode",
        trace_ids=(trace.id,),
    )
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id=trace.id,
    )
    view = vf.ObservationView.capture(
        {"reply": "hello"},
        snapshot_id=source.snapshot_id,
        builder_revision="terminal@1",
        scope="retrospective",
        subjects=(subject,),
    )
    signal = vf.SignalDefinition(
        signal_id="correct",
        revision="1",
        semantics="outcome",
        description="Verified correctness",
        units="indicator",
        minimum=0,
        maximum=1,
    )
    run = vf.AssessmentRun(
        run_id="run",
        producer_id="deterministic",
        producer_revision="1",
        rubric_revision="1",
        snapshot_id=source.snapshot_id,
        invocation_id="invocation",
        attempt_id="attempt",
        expected=(
            vf.AssessmentTarget(
                subject=subject,
                signal=signal,
            ),
        ),
    )
    assessment = vf.Assessment(
        assessment_id="assessment",
        run_id=run.run_id,
        subject=subject,
        view_id=view.view_id,
        signal=signal,
        status="valid",
        value=0,
    )
    batch = vf.AssessmentBatch(
        source=source,
        run=run,
        views=(view,),
        assessments=(assessment,),
    )
    trace.assessment_batches.append(batch)
    assignment_request = vf.CreditRequest(
        source=source,
        invocation_id="assignment-invocation",
        attempt_id="assignment-attempt",
        rule=vf.CreditRule(rule_id="retain-outcome", revision="1"),
        accepted=(assessment,),
        targets=(vf.CreditTarget(recipient=subject, channel="outcome"),),
        allocation="turn_boundary",
        overlap_policy="reject",
    )
    contribution = vf.CreditContribution(
        contribution_id="credit",
        parent_assessment_ids=(assessment.assessment_id,),
        recipient=subject,
        channel="outcome",
        signal=signal,
        transformation="identity",
        status="valid",
        value=0,
        allocation="turn_boundary",
        attribution="coarse",
    )
    assignment = vf.CreditAssignment(
        request=assignment_request, contributions=(contribution,)
    )
    trace.credit_assignments.append(assignment)
    trace.record_reward("legacy", 0.25)
    loaded = vf.WireTrace.model_validate(trace.to_record())
    assert loaded.assessment_batches == [batch]
    assert loaded.credit_assignments == [assignment]
    assert loaded.reward == 0.25
    assert loaded.assessment_batches[0].assessments[0].value == 0
    episode = vf.Episode(
        id="episode",
        task=trace.task,
        traces=[trace],
        assessment_batches=[batch],
        credit_assignments=[assignment],
    )
    loaded_episode = vf.WireEpisode.model_validate(episode.to_record())
    assert loaded_episode.assessment_batches == [batch]
    assert loaded_episode.credit_assignments == [assignment]
    assert loaded_episode.traces[0].credit_assignments == [assignment]
    assert loaded_episode.traces[0].assessment_batches == [batch]
    incomplete_record = trace.to_record()
    incomplete_record.pop("assessment_sources")
    incomplete_record["assessment_batches"][0]["source"] = source.identity.model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="dangling source identity"):
        vf.WireTrace.model_validate(incomplete_record)
    incomplete_assignment = trace.to_record()
    incomplete_assignment.pop("assessment_sources")
    incomplete_assignment["credit_assignments"][0]["request"]["source"] = (
        source.identity.model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="dangling source identity"):
        vf.WireTrace.model_validate(incomplete_assignment)


def test_assessment_projection_uses_original_tokens_and_rejects_rewrites():
    from verifiers.v1.assessment_source import capture_trace_source

    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="hello")),
        nodes=[
            MessageNode(
                parent=None,
                message=vf.UserMessage(content="hello"),
                token_ids=[1],
                mask=[False],
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[2, 3, 4],
                mask=[False, True, True],
            ),
        ],
    )
    source = capture_trace_source(trace)
    subject = vf.SubjectRef(
        kind="turn",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id=trace.id,
        node_index=1,
        node_content_digest=source.nodes[1].node_content_digest,
    )
    result = vf.project_subject(subject, source, trace)
    assert result.status == "exact_turn"
    assert [(span.start, span.end) for span in result.intervals] == [(1, 3)]
    signal = vf.SignalDefinition(
        signal_id="guard",
        revision="1",
        semantics="cost",
        description="harmful action",
        units="reward",
    )
    parent = vf.Assessment(
        assessment_id="parent",
        run_id="run",
        subject=subject,
        view_id="context",
        signal=signal,
        status="valid",
        value=-1,
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="assignment",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="guard", revision="1"),
        accepted=(parent,),
        targets=(vf.CreditTarget(recipient=subject, channel="guard"),),
        allocation="fixed_mass",
        overlap_policy="reject",
    )
    assignment = vf.CreditAssignment(
        request=request,
        contributions=(
            vf.CreditContribution(
                contribution_id="harm",
                parent_assessment_ids=("parent",),
                recipient=subject,
                channel="guard",
                signal=signal,
                transformation="identity",
                status="valid",
                value=-1,
                allocation="fixed_mass",
                attribution="coarse",
            ),
        ),
    )
    aligned = vf.project_assignment(assignment, trace)
    assert aligned.assignment == assignment
    assert aligned.contributions[0].projection == result
    assert vf.CreditAlignment.model_validate_json(aligned.model_dump_json()) == aligned
    outcome_subject = vf.SubjectRef(
        kind="episode",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
    )
    # Whole outcomes need source compatibility, although they need no token mask.
    assert (
        vf.project_subject(outcome_subject, source, trace).reason
        == "explicit_turn_or_span_required"
    )
    changed = trace.model_copy(deep=True)
    changed.record_reward("unrelated", 0.9)
    changed.info["annotation"] = "later annotation"
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "explicit_turn_or_span_required"
    )
    changed.task = changed.task.model_copy(
        update={
            "data": changed.task.data.model_copy(update={"prompt": "different task"}),
        }
    )
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "current_execution_configuration_mismatch"
    )
    changed = trace.model_copy(deep=True)
    changed.nodes.append(MessageNode(parent=1, message=vf.UserMessage(content="next")))
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "whole_source_node_count_changed"
    )
    sibling = trace.model_copy(deep=True)
    sibling.id = "new-sibling"
    assert (
        vf.project_subject(
            outcome_subject, source, {trace.id: trace, sibling.id: sibling}
        ).reason
        == "current_episode_child_manifest_mismatch"
    )
    changed = trace.model_copy(deep=True)
    changed.state.artifacts["new.txt"] = b"new evidence"
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "current_artifacts_changed_or_unavailable"
    )
    changed = trace.model_copy(deep=True)
    changed.tools.append(vf.Tool(name="new_tool", description="new", parameters={}))
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "current_tools_changed"
    )
    changed = trace.model_copy(deep=True)
    changed.calls.append(vf.ModelCall(node=1, model="different-model"))
    assert (
        vf.project_subject(outcome_subject, source, changed).reason
        == "current_calls_changed"
    )
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert vf.project_subject(subject, source, restored) == result
    restored.agent.execution_purpose = "assessment"
    assert (
        vf.project_subject(subject, source, restored).reason
        == "assessment_execution_trace"
    )
    restored.agent.execution_purpose = "solver"
    trace.nodes.append(MessageNode(parent=1, message=vf.UserMessage(content="next")))
    assert vf.project_subject(subject, source, trace) == result
    span = vf.SubjectRef(
        kind="span",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id=trace.id,
        node_index=1,
        node_content_digest=subject.node_content_digest,
        span_start=0,
        span_end=2,
        representation="full_tokens",
        representation_digest=vf.token_representation_digest(trace.nodes[1]),
    )
    assert (
        vf.project_subject(span, source, trace).reason
        == "span_contains_unsampled_tokens"
    )
    trace.nodes[0].message = vf.UserMessage(content="rewritten prompt")
    assert vf.project_subject(subject, source, trace).status == "failed"
    assert (
        vf.project_assignment(assignment, trace).contributions[0].projection.status
        == "failed"
    )


async def _linked_execution_projection_fixture(fidelity="exact", *, linked=True):
    """Qualified fixture coordinates test transport, not production parser fidelity."""
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception.tool import MCPDispatch, ToolHookRequest
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.types import ToolCall, ToolMessage, generated_completion_digest

    tokens = [101, 102, 103, 104, 105]
    mask = [False, True, True, True, True]
    call = ToolCall(id="original-call", name="counter_bump", arguments="{}")
    revision = (
        "unqualified:fixture@1"
        if fidelity == "unqualified"
        else "fixture-original-execution-span@1"
    )
    producer = vf.GeneratedCallProducer.capture(revision, {"kind": "test_fixture"})
    generated = (
        ()
        if fidelity == "missing"
        else (
            vf.GeneratedCallAttempt(
                attempt_index=0,
                emitted_call_index=None,
                parse_status="malformed",
                raw="un-emitted parser attempt",
                coordinate_system="node_local_full_tokens",
                parser_revision=revision,
                span_fidelity="unavailable",
                completion_token_digest=generated_completion_digest(tokens[1:]),
            ),
            vf.GeneratedCallAttempt(
                attempt_index=1,
                emitted_call_index=0,
                provider_call_id=call.id,
                parse_status="parsed",
                raw="counter_bump({})",
                name=call.name,
                arguments="{}",
                coordinate_system="node_local_full_tokens",
                parser_revision=revision,
                token_span=None if fidelity == "unavailable" else (2, 4),
                span_fidelity=fidelity
                if fidelity in {"joint", "unavailable"}
                else "exact",
                completion_token_digest=generated_completion_digest(tokens[1:]),
            ),
        )
    )
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="bump")),
        nodes=[
            MessageNode(
                message=UserMessage(content="bump"), token_ids=[1], mask=[False]
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="", tool_calls=[call]),
                sampled=True,
                token_ids=tokens,
                mask=mask,
                generated_calls=generated,
                generated_call_producer=producer if generated else None,
            ),
        ],
        calls=[vf.ModelCall(node=1, finish_reason="tool_calls")],
    )
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    message = ToolMessage(tool_call_id=call.id, name=call.name, content="")
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
    for transport_attempt in range(2):
        provenance = (
            {
                "parent_execution_id": "parent",
                "dispatch_ticket": decision["mcp_dispatch_ticket"],
                "transport_attempt_index": transport_attempt,
                "server_name": "counter",
            }
            if linked
            else {}
        )
        receipt = vf.ToolServerReceipt(
            invocation_id=f"physical-{transport_attempt}",
            event_index=0,
            phase="dispatch",
            tool_name="bump",
            arguments_json='{"args":[],"kwargs":{}}',
            **provenance,
        )
        session.retain_tool_server_receipt(receipt)
        terminal = receipt.model_copy(
            update={"event_index": 1, "phase": "returned", "result_json": '"observed"'}
        )
        session.retain_tool_server_receipt(terminal)
    after = ToolHookRequest(
        phase="after",
        message=message,
        execution_id="parent",
        call=call,
        event_index=2,
        raw_result="observed",
    )
    await session.handle_tool("after", message, request=after)
    source = capture_trace_source(trace)
    subjects = tuple(
        vf.SubjectRef(
            kind="execution",
            snapshot_id=source.snapshot_id,
            episode_id=source.episode_id,
            trace_id=trace.id,
            execution=ref,
        )
        for ref in source.executions
    )
    return trace, source, subjects


@pytest.mark.asyncio
async def test_linked_executions_project_exact_original_call_and_retry_coordinates():
    trace, source, subjects = await _linked_execution_projection_fixture()
    call = vf.SubjectRef(
        kind="call",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id=trace.id,
        node_index=1,
        node_content_digest=source.nodes[1].node_content_digest,
        call_index=1,
    )
    expected = vf.project_subject(call, source, trace)
    assert expected.status == "exact_call"
    assert [(span.start, span.end) for span in expected.intervals] == [(2, 4)]
    assert len(subjects) == 3
    assert len({subject.subject_id for subject in subjects}) == 3
    assert {subject.execution.invocation_id for subject in subjects} == {
        "parent",
        "physical-0",
        "physical-1",
    }
    restored = vf.WireTrace.model_validate(trace.to_record())
    for subject in subjects:
        projected = vf.project_subject(subject, source, trace)
        assert projected.status == "exact_call"
        assert projected.subject_id == subject.subject_id
        assert projected.intervals == expected.intervals
        assert projected.members == (expected,)
        assert vf.project_subject(subject, source, restored) == projected
    assert trace.nodes[1].token_ids == [101, 102, 103, 104, 105]
    assert trace.nodes[1].mask == [False, True, True, True, True]
    # Transport retries 0 and 1 both map to generated attempt 1 / emitted call 0.
    assert all(
        event.generated_attempt_index == 1
        for event in trace.tool_execution_events
        if event.source == "harness"
    )
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.session import RolloutSession

    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    later = vf.ToolServerReceipt(
        invocation_id="later-unrelated",
        event_index=0,
        phase="dispatch",
        tool_name="other",
        arguments_json="{}",
    )
    session.retain_tool_server_receipt(later)
    session.retain_tool_server_receipt(
        later.model_copy(
            update={"event_index": 1, "phase": "returned", "result_json": '"later"'}
        )
    )
    trace.nodes.append(MessageNode(parent=1, message=UserMessage(content="later")))
    for subject in subjects:
        assert vf.project_subject(subject, source, trace) == vf.project_subject(
            subject, source, restored
        )
    # Projection admits only its sealed occurrence and required parent prefix.
    # A later invalid receipt must not erase already qualified sampled support;
    # complete current-trace admission remains a separate caller responsibility.
    events = list(trace.tool_execution_events)
    assert events[-1].invocation_id == "later-unrelated"
    events[-1] = events[-1].model_copy(update={"event_index": True})
    trace.tool_execution_events = tuple(events)
    for subject in subjects:
        assert vf.project_subject(subject, source, trace) == vf.project_subject(
            subject, source, restored
        )
    with pytest.raises(ValueError):
        type(events[-1]).model_validate(events[-1].model_dump(mode="python"))
    with pytest.raises(ValueError):
        trace.to_record()
    with pytest.raises(ValueError):
        vf.WireTrace.model_validate(trace.model_dump(mode="python"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "missing_return",
        "rewritten_return",
        "missing_parent",
        "changed_tokens",
        "boolean_node",
        "boolean_emitted",
        "boolean_transport_attempt",
    ],
)
async def test_execution_projection_rejects_current_provenance_drift(change):
    trace, source, subjects = await _linked_execution_projection_fixture()
    changed = trace.model_copy(deep=True)
    if change == "changed_tokens":
        changed.nodes[1].token_ids[2] = 999
    elif change == "missing_parent":
        changed.tool_execution_events = tuple(
            event
            for event in changed.tool_execution_events
            if not (event.source == "harness" and event.phase == "dispatch")
        )
    elif change == "missing_return":
        changed.tool_execution_events = tuple(
            event
            for event in changed.tool_execution_events
            if not (
                event.source == "tool_server"
                and event.invocation_id == "physical-0"
                and event.phase == "returned"
            )
        )
    elif change in {"boolean_node", "boolean_emitted"}:
        field, value = (
            ("node_index", True)
            if change == "boolean_node"
            else ("emitted_call_index", False)
        )
        changed.tool_execution_events = tuple(
            event.model_copy(update={field: value})
            if event.source == "harness" and event.phase == "dispatch"
            else event
            for event in changed.tool_execution_events
        )
    else:
        events = list(changed.tool_execution_events)
        index = next(
            index
            for index, event in enumerate(events)
            if event.source == "tool_server"
            and event.invocation_id == "physical-0"
            and event.phase == "returned"
        )
        receipt = json.loads(events[index].receipt_json)
        if change == "boolean_transport_attempt":
            receipt["transport_attempt_index"] = True
        else:
            receipt["result_json"] = '"rewritten"'
        from verifiers.v1.assessments import canonical_json

        events[index] = events[index].model_copy(
            update={"receipt_json": canonical_json(receipt)}
        )
        changed.tool_execution_events = tuple(events)
    subject = next(
        subject
        for subject in subjects
        if subject.execution.invocation_id == "physical-0"
    )
    result = vf.project_subject(subject, source, changed)
    assert result.status == "failed"
    assert not result.intervals
    if change.startswith("boolean_"):
        assert result.reason == "current_tool_execution_events_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["invocation_id", "receipt_json"])
async def test_copied_server_event_bytes_cannot_be_laundered_by_projection_or_export(
    field,
):
    trace, source, subjects = await _linked_execution_projection_fixture()
    original_record = trace.to_record()
    subject = next(
        subject
        for subject in subjects
        if subject.execution.invocation_id == "physical-0"
    )
    expected = vf.project_subject(subject, source, trace)
    assert expected.status == "exact_call"
    restored = vf.WireTrace.model_validate(original_record)
    assert vf.project_subject(subject, source, restored) == expected
    changed = trace.model_copy(deep=True)
    events = list(changed.tool_execution_events)
    index = next(
        index
        for index, event in enumerate(events)
        if event.source == "tool_server"
        and event.invocation_id == "physical-0"
        and event.phase == "returned"
    )
    original = events[index]
    events[index] = original.model_copy(
        update={field: getattr(original, field).encode("utf-8")}
    )
    changed.tool_execution_events = tuple(events)
    with pytest.raises(ValueError):
        changed.validate_tool_execution_events()
    with pytest.raises(ValueError):
        changed.to_record()
    # Pydantic instances may otherwise skip field re-validation at re-admission.
    copied_input = {**original_record, "tool_execution_events": tuple(events)}
    with pytest.raises(ValueError):
        vf.WireTrace.model_validate(copied_input)
    projected = vf.project_subject(subject, source, changed)
    assert projected.status == "failed" and not projected.intervals
    assert projected.reason == "current_tool_execution_events_changed"
    assert trace.to_record() == original_record
    assert vf.project_subject(subject, source, trace) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["invocation_id", "arguments_json"])
async def test_copied_server_receipt_bytes_rejected_before_retention_and_transport(
    field,
):
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.session import RolloutSession

    class Tools(vf.Toolset[vf.ToolsetConfig, vf.State]):
        pass

    trace, source, subjects = await _linked_execution_projection_fixture()
    original_record = trace.to_record()
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    event = next(
        event
        for event in trace.tool_execution_events
        if event.source == "tool_server" and event.phase == "dispatch"
    )
    receipt = vf.ToolServerReceipt.model_validate_json(event.receipt_json)
    forged = receipt.model_copy(update={field: getattr(receipt, field).encode("utf-8")})
    with pytest.raises(ValueError, match="valid string"):
        session.retain_tool_server_receipt(forged)
    with pytest.raises(ValueError, match="valid string"):
        await Tools(vf.ToolsetConfig())._emit_execution_receipt(forged)
    assert trace.to_record() == original_record
    accepted = session.retain_tool_server_receipt(receipt)
    assert accepted["ok"] is True and accepted["receipt_seq"] == event.receipt_seq
    assert trace.to_record() == original_record
    reloaded = vf.WireTrace.model_validate(original_record)
    for subject in subjects:
        assert vf.project_subject(subject, source, reloaded) == vf.project_subject(
            subject, source, trace
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("node_index", True), ("execution_id", b"parent")]
)
async def test_direct_source_capture_rejects_copied_harness_coordinate_types(
    field, value
):
    from verifiers.v1.assessment_source import capture_trace_source

    trace, source, _ = await _linked_execution_projection_fixture()
    unchanged = trace.to_record()
    changed = trace.model_copy(deep=True)
    changed.tool_execution_events = tuple(
        event.model_copy(update={field: value})
        if event.source == "harness" and event.phase == "dispatch"
        else event
        for event in changed.tool_execution_events
    )
    with pytest.raises(ValueError):
        capture_trace_source(changed)
    with pytest.raises(ValueError):
        changed.to_record()
    assert trace.to_record() == unchanged
    assert capture_trace_source(trace) == source


@pytest.mark.parametrize(
    "field,value",
    [
        ("write_id", b"write"),
        ("body_digest", b"a" * 64),
        ("expected_revision", False),
        ("applied_revision", True),
        ("conflict", 0),
    ],
)
def test_direct_source_capture_rejects_copied_state_write_types(field, value):
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.trace import StateWriteReceipt

    write = StateWriteReceipt(
        write_id="write",
        body_digest="a" * 64,
        expected_revision=0,
        applied_revision=1,
        conflict=False,
    )
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="write")),
        state_write_receipts=(write,),
        tool_state_revision=1,
    )
    source = capture_trace_source(trace)
    original = trace.to_record()
    changed = trace.model_copy(
        update={"state_write_receipts": (write.model_copy(update={field: value}),)}
    )
    with pytest.raises(ValueError):
        capture_trace_source(changed)
    with pytest.raises(ValueError):
        changed.to_record()
    assert capture_trace_source(vf.WireTrace.model_validate(original)) == source
    assert trace.to_record() == original


@pytest.mark.asyncio
async def test_copied_hook_execution_id_bytes_rejected_before_ticket_replay():
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception.tool import ToolHookRequest
    from verifiers.v1.session import RolloutSession

    trace, _, _ = await _linked_execution_projection_fixture()
    original = trace.to_record()
    dispatch = next(
        event
        for event in trace.tool_execution_events
        if event.source == "harness" and event.phase == "dispatch"
    )
    hook = ToolHookRequest.model_validate_json(dispatch.request_json)
    assert hook.execution_id == "parent"
    forged = hook.model_copy(update={"execution_id": b"parent"})
    session = RolloutSession(ModelContext("model", EvalClientConfig()), trace)
    with pytest.raises(ValueError, match="valid string"):
        await session.handle_tool("dispatch", forged.message, request=forged)
    assert trace.to_record() == original
    decision = await session.handle_tool("dispatch", hook.message, request=hook)
    assert decision == json.loads(dispatch.decision_json)
    assert decision["mcp_dispatch_ticket"]
    assert trace.to_record() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("fidelity", ["missing", "joint", "unavailable", "unqualified"])
async def test_execution_projection_has_no_fallback_without_exact_call_span(fidelity):
    trace, source, subjects = await _linked_execution_projection_fixture(fidelity)
    # A usable full turn exists; execution projection must not silently use it.
    turn = vf.SubjectRef(
        kind="turn",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id=trace.id,
        node_index=1,
        node_content_digest=source.nodes[1].node_content_digest,
    )
    assert vf.project_subject(turn, source, trace).status == "exact_turn"
    for subject in subjects:
        result = vf.project_subject(subject, source, trace)
        assert result.status == "unsupported"
        assert result.subject_id == subject.subject_id
        assert not result.intervals


@pytest.mark.asyncio
async def test_unlinked_server_execution_never_guesses_parent_from_equal_content():
    trace, source, subjects = await _linked_execution_projection_fixture(linked=False)
    for subject in subjects:
        result = vf.project_subject(subject, source, trace)
        if subject.execution.origin == "tool_server":
            assert result.status == "unsupported" and not result.intervals
        else:
            assert result.status == "exact_call"


def test_agent_execution_purpose_is_explicit_and_observed_before_callbacks():
    from verifiers.v1.agent import Agent, _EpisodeAgent

    def trace():
        return vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        )

    assert trace().agent.execution_purpose == "solver"
    legacy = trace().to_record()
    legacy["agent"].pop("execution_purpose")
    legacy["agent"]["name"] = "judge"
    legacy["agent"]["trainable"] = False
    assert vf.Trace.model_validate(legacy).agent.execution_purpose == "solver"
    invalid = trace().to_record()
    invalid["agent"]["execution_purpose"] = "guessed_from_name"
    with pytest.raises(ValueError):
        vf.Trace.model_validate(invalid)
    for purpose, trainable in (
        ("solver", False),
        ("assessment", False),
        ("assessment", True),
    ):
        observed = []
        standalone = object.__new__(Agent)
        standalone.trainable = trainable
        standalone.execution_purpose = purpose
        current = trace()
        standalone._watch_standing(
            lambda item, observed=observed: observed.append(
                (item.agent.execution_purpose, item.agent.trainable)
            )
        )(current)
        assert observed == [(purpose, trainable)]
        restored = vf.Trace.model_validate(current.to_record())
        assert (restored.agent.execution_purpose, restored.agent.trainable) == (
            purpose,
            trainable,
        )

        seat = object.__new__(_EpisodeAgent)
        seat.trainable, seat.execution_purpose = trainable, purpose
        # A name does not classify execution purpose.
        seat._name = "judge" if purpose == "solver" else "solver"
        seat._on_discard = None
        seat._on_trace = lambda item, observed=observed: observed.append(
            (item.agent.execution_purpose, item.agent.trainable)
        )
        seat._watch(
            lambda item, observed=observed: observed.append(
                (item.agent.execution_purpose, item.agent.trainable)
            )
        )(trace())
        assert observed == [(purpose, trainable)] * 3


@pytest.mark.asyncio
async def test_episode_assessment_keeps_cross_trace_and_zero_child_evidence():
    import json

    class AssessedEnv(vf.Env):
        def __init__(self, fail=False):
            self.fail = fail
            self.plan_context = False
            self.fail_assessor = False
            self.sequence = 0
            self.contexts = []
            self.config = SimpleNamespace(
                env_id="test", timeout=SimpleNamespace(episode=1, finalize=1)
            )

        def _episode_agents(self, ctx, completed, on_trace, on_discard):
            self.completed, self.on_trace = completed, on_trace

        async def run(self, task, agents):
            if self.fail:
                raise ValueError("execution failed")
            for _ in range(2):
                trace = vf.Trace(
                    agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(type="Task", data=task.data),
                    ok=True,
                )
                self.on_trace(trace)
                self.completed.append(trace)

        def assessment_requests(self, source):
            self.sequence += 1
            subject = vf.SubjectRef(
                kind="episode",
                snapshot_id=source.snapshot_id,
                episode_id=source.episode_id,
            )
            view = vf.ObservationView.capture(
                json.loads(source.source_json)["execution"],
                snapshot_id=source.snapshot_id,
                builder_revision="execution@1",
                scope="retrospective",
                subjects=(subject,),
            )
            signal = vf.SignalDefinition(
                signal_id="execution_ok",
                revision="1",
                semantics="outcome",
                description="Execution completed",
                units="indicator",
            )
            run = vf.AssessmentRun(
                run_id=f"run-{self.sequence}",
                producer_id="execution",
                producer_revision="1",
                rubric_revision="1",
                snapshot_id=source.snapshot_id,
                invocation_id=f"invocation-{self.sequence}",
                attempt_id=f"attempt-{self.sequence}",
                expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
            )
            return [
                (
                    "execution",
                    vf.AssessmentRequest(
                        source=source.identity, run=run, views=(view,)
                    ),
                )
            ]

        @vf.assessment
        async def execution(self, request, ctx):
            if self.fail_assessor:
                raise RuntimeError("current episode assessor failed")
            target = request.run.expected[0]
            yield vf.Assessment(
                assessment_id=f"result-{request.run.attempt_id}",
                run_id=request.run.run_id,
                subject=target.subject,
                signal=target.signal,
                view_id=request.views[0].view_id,
                status="valid",
                value=int(ctx.input(request.views[0].view_id)["ok"]),
            )

        def plan_credit(self, source, assessments, context):
            if not self.plan_context:
                return super().plan_credit(source, assessments, context)
            self.contexts.append(context)
            current = {
                (run.run_id, run.attempt_id, run.invocation_id)
                for run in context.current_assessment_runs
            }
            selected = tuple(
                batch
                for batch in assessments
                if (batch.run.run_id, batch.run.attempt_id, batch.run.invocation_id)
                in current
                and batch.run.status == "complete"
            )
            if not selected or any(
                item.status == "complete" for item in context.prior_assignments
            ):
                return []
            return self.credit_requests(source, selected)

        def credit_requests(self, source, assessments):
            accepted = assessments[-1].assessments
            return [
                (
                    "execution_credit",
                    vf.CreditRequest(
                        source=source,
                        invocation_id="credit-invocation",
                        attempt_id="credit-attempt",
                        rule=vf.CreditRule(rule_id="execution-outcome", revision="1"),
                        accepted=accepted,
                        targets=(vf.CreditTarget(recipient=accepted[0].subject),),
                        allocation="turn_boundary",
                        overlap_policy="reject",
                    ),
                )
            ]

        @vf.credit
        async def execution_credit(self, request):
            parent = request.accepted[0]
            yield vf.CreditContribution(
                contribution_id="credit",
                parent_assessment_ids=(parent.assessment_id,),
                recipient=parent.subject,
                signal=parent.signal,
                transformation="identity",
                status="valid",
                value=parent.value,
                allocation="turn_boundary",
                attribution="coarse",
            )

    task = vf.Task(vf.TaskData(prompt="hello"))
    for failure in (False, True):
        episode = await AssessedEnv(failure).run_episode(task, None)
        assert episode.ok is not failure
        assert not episode.credit_errors
        assert episode.credit_assignments[-1].status == "complete"
        assert episode.credit_assignments[-1].contributions[0].value == int(not failure)
        assert len(episode.traces) == (0 if failure else 2)
        assert episode.assessment_finalization_state == (
            "not_run" if failure else "completed"
        )
        terminal = [
            batch
            for batch in episode.assessment_batches
            if batch.run.status == "complete"
        ]
        assert len(terminal) == 1
        assert terminal[0].assessments[0].value == int(not failure)
        assert all(not trace.assessment_batches for trace in episode.traces)
        restored = vf.WireEpisode.model_validate(episode.to_record())
        assert restored.assessment_batches == episode.assessment_batches
        assert len(restored.errors) == int(failure)

    env = AssessedEnv()
    env.plan_context = True
    episode = await env.run_episode(task, None)
    assignments = list(episode.credit_assignments)
    first = env.contexts[0]
    assert len(first.current_assessment_runs) == 1 and not first.prior_assignments
    await env.score_assessments(task, episode, finalization_state="completed")
    assert episode.credit_assignments == assignments
    assert env.contexts[-1].prior_assignments == tuple(assignments)
    env.fail_assessor = True
    await env.score_assessments(task, episode, finalization_state="completed")
    current = {run.run_id for run in env.contexts[-1].current_assessment_runs}
    assert all(
        batch.run.status != "complete"
        for batch in episode.assessment_batches
        if batch.run.run_id in current
    )
    assert any(batch.run.status == "complete" for batch in episode.assessment_batches)
    assert episode.credit_assignments == assignments and not episode.credit_errors


def test_bare_trace_round_trip():
    # The minimal trace: a base task, no nodes, no extras — dump and back into a plain Trace.
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=3, prompt="hello"),
            key="dataset/example-3",
            hash="content-digest",
        ),
    )
    rt = vf.Trace.model_validate(tr.model_dump())
    assert rt.id == tr.id
    assert rt.task.type == "Task"
    assert rt.task.data.idx == 3 and rt.task.data.prompt == "hello"
    assert rt.task.key == "dataset/example-3" and rt.task.hash == "content-digest"
    assert rt.num_turns == 0 and rt.num_branches == 0
    assert rt.reward == 0.0 and rt.errors == []


def test_custom_task_state_round_trip():
    # Custom data and state round-trip into the same parameterization. Data fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[MyTask, MyState](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="gold")),
        state=MyState(score=7),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.model_dump()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[MyTask, MyState].model_validate(wire)
    assert (
        isinstance(rt.task.data, MyTask) and rt.task.data.answer == "gold"
    )  # typed custom field
    assert rt.task.type == "MyTask"  # the producing class's name survives the wire
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5  # property recomputed from `rewards`


def test_wire_trace_round_trip():
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[MyTask, vf.State](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="a")),
        tools=[vf.Tool(name="echo", description="", parameters={"type": "object"})],
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.rewards.setdefault("solved", None)  # seeded: expected but never scored
    tr.metrics.setdefault("acc", None)
    tr.info = {"build": "ok"}
    tr.root_reply = "root answer"
    tr.stop("done")

    # the dump is plain pydantic — derived values are properties, so they're not serialized
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert "reward" not in data and "is_truncated" not in data
    # exclude_none drops None FIELDS, not None dict values — unscored seeds survive
    assert data["rewards"]["solved"] is None and data["metrics"]["acc"] is None

    rt = vf.WireTrace.model_validate(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # property recomputed from `rewards`, seeds contribute 0
    assert rt.rewards["solved"] is None
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.root_reply == "root answer"
    assert rt.last_reply == "root answer"
    rt.root_reply = ""
    assert rt.last_reply == ""
    assert (
        rt.tools == tr.tools
    )  # the advertised tools persist (tool-use SFT reads them)
    assert rt.task.data.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTaskData

    # the env-server wire form (a plain model_dump) loads too
    assert vf.WireTrace.model_validate(tr.model_dump()).num_branches == 2


def _semantic_edge_set() -> vf.SemanticEdgeSet:
    return vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-compact",
                type="continuation",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="child-turn",
                type="subagent_call",
            ),
            vf.SemanticEdge(
                source_request_id="child-turn",
                target_request_id="root-after",
                type="subagent_return",
            ),
            vf.SemanticEdge(
                source_request_id="root-compact",
                target_request_id="root-after",
                type="compaction",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-after",
                type="critic_review",
            ),
        ],
    )


def test_semantic_edges_resolve_to_message_nodes_and_round_trip():
    """Request edges resolve by exact IDs, not call adjacency or graph shape."""
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="root turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="child")),
            MessageNode(
                parent=2, message=AssistantMessage(content="child turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="summarize")),
            MessageNode(
                parent=4, message=AssistantMessage(content="summary"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="resume")),
            MessageNode(
                parent=6, message=AssistantMessage(content="done"), sampled=True
            ),
        ],
    )
    tr.calls = [
        vf.ModelCall(
            node=1,
            acp=vf.ACPInfo(request_id="root-turn"),
        ),
        vf.ModelCall(
            node=3,
            acp=vf.ACPInfo(request_id="child-turn"),
        ),
        vf.ModelCall(
            node=5,
            acp=vf.ACPInfo(request_id="root-compact"),
        ),
        vf.ModelCall(
            node=7,
            acp=vf.ACPInfo(request_id="root-after"),
        ),
    ]

    edge_set = _semantic_edge_set()
    tr.add_semantic_edges(vf.SemanticEdgeSet(edges=edge_set.edges[:2]))
    first_semantic_parents = tr.nodes[3].semantic_parents
    tr.add_semantic_edges(vf.SemanticEdgeSet.model_validate(edge_set.model_dump()))
    expected_parents = [
        [],
        [],
        [],
        [vf.ParentLink(node=1, type="subagent_call")],
        [],
        [vf.ParentLink(node=1, type="continuation")],
        [],
        [
            vf.ParentLink(node=3, type="subagent_return"),
            vf.ParentLink(node=5, type="compaction"),
            vf.ParentLink(node=1, type="critic_review"),
        ],
    ]
    assert [node.semantic_parents for node in tr.nodes] == expected_parents
    assert tr.nodes[3].semantic_parents is first_semantic_parents

    restored = vf.WireTrace.model_validate_json(tr.model_dump_json())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents
    assert [call.acp for call in restored.calls] == [call.acp for call in tr.calls]

    # The base ACP layer resolves the generic edge set before harness-owned metadata.
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    turn_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, turn_metadata)
    harness.acp_turn_result(
        restored, vf.ACPTurn(reply="done", response_metadata=turn_metadata)
    )
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # session/close may publish the same cumulative edge set again.
    close_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, close_metadata)
    harness.acp_close_result(restored, close_metadata)
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # A failed provider exchange and its SDK retry share one logical request ID.
    restored.calls.append(
        vf.ModelCall(
            acp=restored.calls[0].acp,
            error=vf.Error(type="E", message="x"),
        )
    )
    restored.add_semantic_edges(_semantic_edge_set())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents


def test_semantic_edge_uses_last_committed_retry_node():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 1"), sampled=True
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 2"), sampled=True
            ),
            MessageNode(
                parent=None, message=AssistantMessage(content="next"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=2, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="next")),
        ],
    )

    tr.add_semantic_edges(
        vf.SemanticEdgeSet(
            edges=[
                vf.SemanticEdge(
                    source_request_id="retried",
                    target_request_id="next",
                    type="continuation",
                )
            ]
        )
    )

    assert tr.nodes[3].semantic_parents == [vf.ParentLink(node=2, type="continuation")]


def test_semantic_edge_cycle_is_rejected_without_partial_mutation():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="start")),
            MessageNode(
                parent=0, message=AssistantMessage(content="first"), sampled=True
            ),
            MessageNode(parent=1, message=UserMessage(content="continue")),
            MessageNode(
                parent=2, message=AssistantMessage(content="second"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="first")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="second")),
        ],
    )

    with pytest.raises(ValueError, match="cycle in the message graph"):
        tr.add_semantic_edges(
            vf.SemanticEdgeSet(
                edges=[
                    vf.SemanticEdge(
                        source_request_id="second",
                        target_request_id="first",
                        type="custom",
                    )
                ]
            )
        )

    assert all(not node.semantic_parents for node in tr.nodes)


def test_acp_info_is_validated_and_stripped():
    headers = {
        "Authorization": "Bearer local",
        "Idempotency-Key": "provider-key",
        "X-ACP-Model-Request-ID": "request-1",
        "OpenAI-Beta": "feature",
    }
    acp, forwarded = extract_acp_info(headers)
    assert acp == vf.ACPInfo(request_id="request-1")
    assert not ACP_EXTENSION_HEADERS.intersection(map(str.lower, forwarded))
    assert forwarded["Idempotency-Key"] == "provider-key"
    assert forwarded["OpenAI-Beta"] == "feature"

    absent, unchanged = extract_acp_info({"OpenAI-Beta": "feature"})
    assert absent is None and unchanged == {"OpenAI-Beta": "feature"}

    with pytest.raises(ValueError, match="not a valid ACP request ID"):
        extract_acp_info({"X-ACP-Model-Request-ID": "not/a/valid/id"})


def test_acp_semantic_edge_metadata_is_optional():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    harness._consume_protocol_metadata(trace, {})

    assert all(not node.semantic_parents for node in trace.nodes)

    harness._consume_protocol_metadata(
        trace, {ACP_SEMANTIC_EDGES_METADATA_KEY: {"edges": []}}
    )

    assert all(not node.semantic_parents for node in trace.nodes)


def test_acp_derives_compaction_attempt_branch_trainability():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="work")),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="working"),
                sampled=True,
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
            ),
            MessageNode(parent=1, message=UserMessage(content="summarize")),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="bad tool call"),
                sampled=True,
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-0.2, -0.3],
            ),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="accepted summary"),
                sampled=True,
                token_ids=[4, 5],
                mask=[True, True],
                logprobs=[-0.4, -0.5],
            ),
            MessageNode(parent=0, message=UserMessage(content="compacted context")),
            MessageNode(
                parent=5,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[6],
                mask=[True],
                logprobs=[-0.6],
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="work")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="rejected")),
            vf.ModelCall(node=4, acp=vf.ACPInfo(request_id="accepted")),
            vf.ModelCall(node=6, acp=vf.ACPInfo(request_id="resumed")),
        ],
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                ]
            },
        },
    )

    attempts = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert attempts["bad tool call"].trainable is False
    assert attempts["accepted summary"].trainable is False

    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "accepted",
                        "target_request_id": "resumed",
                        "type": "compaction",
                    },
                ]
            },
        },
    )

    assert trace.nodes[3].sampled is True
    assert trace.nodes[3].mask == [True, True]
    assert trace.nodes[4].mask == [True, True]
    assert trace.nodes[6].mask == [True]
    assert trace.nodes[3].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[4].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[6].semantic_parents == [vf.ParentLink(node=4, type="compaction")]
    assert trace.num_branches == 3
    branches = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert branches["bad tool call"].trainable is False
    assert branches["accepted summary"].trainable is True
    assert branches["answer"].trainable is True
    assert branches["bad tool call"].nodes[-2] is trace.nodes[2]
    assert branches["accepted summary"].nodes[-2] is trace.nodes[2]

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.nodes[3].sampled is True
    assert restored.nodes[3].mask == [True, True]
    assert restored.nodes[4].mask == [True, True]
    restored_branches = {
        branch.nodes[-1].message.content: branch for branch in restored.branches
    }
    assert restored_branches["bad tool call"].trainable is False
    assert restored_branches["accepted summary"].trainable is True


def test_semantic_edge_set_rejects_duplicate_self_and_cyclic_edges():
    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(edge_set["edges"][0])
    with pytest.raises(ValueError, match="duplicate semantic edge"):
        vf.SemanticEdgeSet.model_validate(edge_set)

    with pytest.raises(ValueError, match="cannot link a request to itself"):
        vf.SemanticEdgeSet.model_validate(
            {
                "edges": [
                    {
                        "source_request_id": "request-1",
                        "target_request_id": "request-1",
                        "type": "custom",
                    }
                ]
            }
        )

    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(
        {
            "source_request_id": "root-after",
            "target_request_id": "root-turn",
            "type": "custom",
        }
    )
    with pytest.raises(ValueError, match="semantic edge cycle"):
        vf.SemanticEdgeSet.model_validate(edge_set)


def test_semantic_edge_set_accepts_deep_acyclic_chain():
    edge_set = vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id=f"request-{index}",
                target_request_id=f"request-{index + 1}",
                type="continuation",
            )
            for index in range(2_000)
        ]
    )

    assert len(edge_set.edges) == 2_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "ordinary",
        "deadline",
        "setup",
        "empty",
        "assessor_failure",
        "assessment_timeout",
        "cancel",
        "borrowed",
    ],
)
async def test_failed_rollout_assessment_preserves_original_failure_and_cleanup(mode):
    from verifiers.v1.assessment_source import capture_trace_source

    events = []

    async def cleanup(trace, runtime):
        events.append("cleanup")

    async def stop():
        events.append("stop")

    class PrefixTask:
        scoring_deferred = False
        data = vf.TaskData(idx=0, prompt="task")

        def hooks(self, name):
            assert name == "assessment"
            return ["declared"]

        async def score_assessments(self, trace):
            events.append("assessment")
            source = capture_trace_source(trace)
            assert (
                json.loads(source.source_json)["execution"]["finalization_state"]
                == "not_run"
            )
            if mode == "assessor_failure":
                raise ValueError("source unavailable")
            if mode == "assessment_timeout":
                await asyncio.sleep(10)
            if mode == "cancel":
                raise asyncio.CancelledError

        async def score(self, *args):
            pytest.fail("failed prefix must not run legacy scoring")

        async def finalize(self, *args):
            pytest.fail("failed prefix must not run finalization")

    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="task")),
        nodes=[]
        if mode == "empty"
        else [
            vf.MessageNode(message=vf.AssistantMessage(content="partial"), sampled=True)
        ],
    )
    runtime = SimpleNamespace(stop=stop, stopped=False)
    run = object.__new__(Rollout)
    run.trace, run.task, run.runtime = trace, PrefixTask(), runtime
    run.harness = SimpleNamespace(cleanup=cleanup)
    run._stack = AsyncExitStack()
    run._closed = False
    run._opened = True
    run._failed = False
    run._failure = None
    run._setup_completed = mode != "setup"
    run._assessment_deadline_expired = mode == "deadline"
    run._harness_session = None
    run._borrowed_runtime = runtime if mode == "borrowed" else None
    run._timeouts = RolloutTimeouts(scoring=0.01)
    original = vf.ProviderError("original failure")
    run.fail(original)
    original_errors = list(trace.errors)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run.close()
    else:
        assert await run.close() is trace
    assert run.failure is original
    assert trace.errors == original_errors
    assert trace.rewards == {} and not trace.ok and trace.is_completed
    assert trace.assessment_finalization_state == "not_run"
    assert events.count("assessment") == (
        0 if mode in {"deadline", "setup", "empty"} else 1
    )
    assert events.count("cleanup") == 1
    assert events.count("stop") == (0 if mode == "borrowed" else 1)
    previous = list(events)
    assert await run.close() is trace
    assert events == previous
    if mode in {"deadline", "setup", "empty", "assessor_failure", "assessment_timeout"}:
        assert len(trace.assessment_errors) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,status,reason",
    [
        ("sampled_integer", "unsupported", "not_sampled_assistant"),
        ("mask_integer", "failed", "sampled_mask_type_mismatch"),
        ("token_boolean", "failed", "original_token_type_mismatch"),
        ("token_negative", "failed", "original_token_type_mismatch"),
    ],
)
async def test_direct_harness_projection_rejects_self_consistent_copied_node_types(
    change, status, reason
):
    from verifiers.v1.assessment_source import capture_trace_source

    trace, _, _ = await _linked_execution_projection_fixture(linked=False)
    node = trace.nodes[1]
    if change == "sampled_integer":
        update = {"sampled": 1}
    elif change == "mask_integer":
        update = {
            "mask": [1 if index == 2 else value for index, value in enumerate(node.mask)]
        }
    else:
        # An unsampled prefix token keeps the generated completion digest intact.
        update = {
            "token_ids": [
                (True if change == "token_boolean" else -1) if index == 0 else value
                for index, value in enumerate(node.token_ids)
            ]
        }
    trace.nodes[1] = node.model_copy(update=update)
    source = capture_trace_source(trace)
    parent = next(ref for ref in source.executions if ref.origin == "harness")
    subject = vf.SubjectRef(
        kind="execution",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id=trace.id,
        execution=parent,
    )
    result = vf.project_subject(subject, source, trace)
    assert result.subject_id == subject.subject_id
    assert result.status == status and result.reason == reason
    assert not result.intervals


@pytest.mark.asyncio
async def test_direct_harness_projection_accepts_original_typed_node_control():
    trace, source, subjects = await _linked_execution_projection_fixture(linked=False)
    subject = next(item for item in subjects if item.execution.origin == "harness")
    result = vf.project_subject(subject, source, trace)
    assert result.subject_id == subject.subject_id
    assert result.status == "exact_call"
    assert [(interval.start, interval.end) for interval in result.intervals] == [(2, 4)]
