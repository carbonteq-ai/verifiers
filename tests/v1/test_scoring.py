import asyncio
import copy
import json

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


def _execution_source(*, returned=True, second=False, result=False):
    from verifiers.v1.mcp.execution import ToolServerReceipt
    from verifiers.v1.trace import ToolServerExecutionEvent

    events = []
    for invocation in ["uuid-1", "uuid-2"] if second else ["uuid-1"]:
        for index, phase in enumerate(
            ["dispatch", "returned"] if returned else ["dispatch"]
        ):
            receipt = ToolServerReceipt(
                invocation_id=invocation,
                event_index=index,
                phase=phase,
                tool_name="send",
                arguments_json='{"to":"a"}',
                result_json=json.dumps({"success": result}, separators=(",", ":"))
                if index
                else None,
            )
            events.append(
                ToolServerExecutionEvent(
                    invocation_id=invocation,
                    event_index=index,
                    phase=phase,
                    receipt_seq=len(events),
                    state_revision=0,
                    receipt_json=vf.assessments.canonical_json(
                        receipt.model_dump(mode="json")
                    ),
                ).model_dump(mode="json")
            )
    payload = {"trace_id": "trace", "tool_execution_events": events}
    refs = vf.execution_refs(payload, episode_id="episode", trace_id="trace")
    return vf.SourceSnapshot.capture(
        payload, episode_id="episode", trace_ids=("trace",), executions=refs
    )


def _execution_subject(source, ref=None):
    ref = ref or source.executions[0]
    return vf.SubjectRef(
        kind="execution",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id=ref.trace_id,
        execution=ref,
    )


def test_execution_source_identity_prefix_and_reload():
    from verifiers.v1.assessments import canonical_json, content_digest

    dispatch = _execution_source(returned=False)
    complete = _execution_source()
    first, last = dispatch.executions[0], complete.executions[0]
    assert first.occurrence_id == last.occurrence_id
    assert first.prefix_digest != last.prefix_digest
    assert dispatch.snapshot_id != complete.snapshot_id
    assert complete.schema_version == 2
    reloaded = vf.SourceSnapshot.model_validate_json(complete.model_dump_json())
    assert reloaded == complete
    assert '"arguments_json"' not in reloaded.identity.model_dump_json()
    assert vf.resolve_execution(reloaded, first) == vf.resolve_execution(
        dispatch, first
    )
    # The assessed completed occurrence can have a strictly earlier observed prefix.
    subject = _execution_subject(complete)
    prefix = vf.capture_execution_view(
        complete, (subject,), scope="prefix", observed=(first,)
    )
    assert "result_json" in prefix.input_json  # dispatch original receipt has null only
    material = json.loads(prefix.input_json)
    assert json.loads(material[0]["events"][0]["receipt_json"])["result_json"] is None
    assert len(material[0]["events"]) == 1
    assert vf.ObservationView.model_validate_json(prefix.model_dump_json()) == prefix
    with pytest.raises(ValueError, match="terminal"):
        vf.capture_execution_view(complete, (subject,), scope="prefix")
    with pytest.raises(ValueError, match="terminal"):
        vf.capture_execution_view(
            dispatch, (_execution_subject(dispatch),), scope="through_action_results"
        )
    result_view = vf.capture_execution_view(
        complete, (subject,), scope="through_action_results"
    )
    assert "false" in result_view.input_json  # observed return is not domain success
    # Legacy source IDs retain the original hash formula.
    old = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    assert old.schema_version == 1
    assert old.snapshot_id == content_digest(
        {
            "episode_id": "episode",
            "source": content_digest({}),
            "nodes": [],
            "trace_ids": ("trace",),
        }
    )
    assert "executions" not in old.model_dump(mode="json")
    assert "executions" not in old.identity.model_dump(mode="json")
    legacy_subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=old.snapshot_id,
        episode_id="episode",
        trace_id="trace",
    )
    assert "execution" not in legacy_subject.model_dump(mode="json")
    assert legacy_subject.subject_id == content_digest(
        legacy_subject.model_dump(mode="json")
    )
    working = vf.resolve_execution(complete, last)
    working[0]["phase"] = "tampered"
    assert vf.resolve_execution(complete, last)[0]["phase"] == "dispatch"
    changed = complete.model_dump(mode="json")
    raw = json.loads(changed["source_json"])
    raw["tool_execution_events"][1]["receipt_json"] = raw["tool_execution_events"][1][
        "receipt_json"
    ].replace("false", "true")
    changed["source_json"] = canonical_json(raw)
    with pytest.raises(ValueError, match="digest"):
        vf.SourceSnapshot.model_validate(changed)
    # Even recapturing changed raw material cannot retain a stale prefix index.
    with pytest.raises(ValueError, match="prefix"):
        vf.SourceSnapshot.capture(
            raw,
            episode_id="episode",
            trace_ids=("trace",),
            executions=complete.executions,
        )


def test_execution_membership_distinct_occurrences_and_unsupported_projection():
    source = _execution_source(second=True)
    left, right = (_execution_subject(source, ref) for ref in source.executions)
    assert left.execution.occurrence_id != right.execution.occurrence_id
    from verifiers.v1.credit import _check_subject, _semantic_overlap

    assert not _semantic_overlap(
        left, right
    )  # identical arguments, distinct executions
    assert _semantic_overlap(left, left)
    assert _semantic_overlap(
        vf.SubjectRef(
            kind="group",
            snapshot_id=source.snapshot_id,
            episode_id="episode",
            members=(left, right),
        ),
        left,
    )
    _check_subject(left, source.identity)  # no graph nodes needed
    projection = vf.project_subject(left, source, {})
    assert projection.status == "unsupported"
    assert projection.reason == "execution_generated_call_token_relation_unqualified"
    assert projection.intervals == ()
    absent = left.execution.model_copy(update={"invocation_id": "absent"})
    with pytest.raises(ValueError, match="absent"):
        _check_subject(_execution_subject(source, absent), source)
    collision = left.execution.model_copy(update={"origin": "harness"})
    assert collision.occurrence_id != left.execution.occurrence_id
    with pytest.raises(ValueError, match="member"):
        vf.resolve_execution(source, collision)
    with pytest.raises(ValueError, match="duplicate"):
        vf.SourceSnapshot.capture(
            json.loads(source.source_json),
            episode_id="episode",
            trace_ids=("trace",),
            executions=(source.executions[0], source.executions[0]),
        )
    payload = json.loads(source.source_json)
    payload["tool_execution_events"][1]["invocation_id"] = "wrong"
    with pytest.raises(ValueError):
        vf.execution_refs(payload, episode_id="episode", trace_id="trace")
    missing_dispatch = json.loads(source.source_json)
    missing_dispatch["tool_execution_events"] = missing_dispatch[
        "tool_execution_events"
    ][1:2]
    missing_dispatch["tool_execution_events"][0]["receipt_seq"] = 0
    with pytest.raises(ValueError, match="missing or conflicting"):
        vf.execution_refs(missing_dispatch, episode_id="episode", trace_id="trace")


def test_execution_domain_assignment_without_tokens():
    source = _execution_source()
    subject = _execution_subject(source)
    signal = vf.SignalDefinition(
        signal_id="harm",
        revision="1",
        semantics="other",
        description="Domain violation independently observed",
        units="violations",
    )
    assessment = vf.Assessment(
        assessment_id="finding",
        run_id="run",
        subject=subject,
        view_id="view",
        signal=signal,
        status="valid",
        value=1,
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="rule",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="harm", revision="1"),
        accepted=(assessment,),
        targets=(vf.CreditTarget(recipient=subject),),
        allocation="fixed_mass",
        overlap_policy="reject",
    )
    contribution = vf.CreditContribution(
        contribution_id="credit",
        parent_assessment_ids=("finding",),
        recipient=subject,
        signal=signal,
        transformation="identity",
        status="valid",
        value=1,
        allocation="fixed_mass",
        attribution="exact",
    )
    assignment = vf.CreditAssignment(request=request, contributions=(contribution,))
    assert (
        vf.CreditAssignment.model_validate_json(assignment.model_dump_json())
        == assignment
    )
    alignment = vf.project_assignment(assignment, {})
    assert alignment.contributions[0].projection.status == "unsupported"
    # Broad trace credit and local occurrence credit overlap even without tokens.
    broad = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id="trace",
    )
    broad_target = vf.CreditTarget(recipient=broad)
    broad_contribution = contribution.model_copy(
        update={"contribution_id": "broad", "recipient": broad}
    )
    overlapping_request = request.model_copy(
        update={"targets": (*request.targets, broad_target)}
    )
    with pytest.raises(ValueError, match="overlapping semantic"):
        vf.CreditAssignment(
            request=overlapping_request,
            contributions=(contribution, broad_contribution),
        )
    explicit_sum = overlapping_request.model_copy(update={"overlap_policy": "sum"})
    vf.CreditAssignment(
        request=explicit_sum, contributions=(contribution, broad_contribution)
    )


async def test_execution_assessor_visibility_and_source_validation():
    from verifiers.v1.assessment_runtime import execute_assessment

    complete = _execution_source()
    dispatch = _execution_source(returned=False)
    subject = _execution_subject(complete)
    view = vf.capture_execution_view(
        complete, (subject,), scope="prefix", observed=dispatch.executions
    )
    signal = vf.SignalDefinition(
        signal_id="guard",
        revision="1",
        semantics="other",
        description="Dispatch-only guard",
        units="violations",
    )
    run = vf.AssessmentRun(
        run_id="run",
        producer_id="deterministic",
        producer_revision="1",
        rubric_revision="1",
        snapshot_id=complete.snapshot_id,
        invocation_id="judge",
        attempt_id="attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
    )
    request = vf.AssessmentRequest(source=complete.identity, run=run, views=(view,))

    async def assess(request, context):
        assert '"result_json"' not in request.source.model_dump_json()
        with pytest.raises(ValueError, match="retrospective"):
            context.retrospective_source()
        observed = json.loads(context.views[0].input_json)
        assert len(observed[0]["events"]) == 1
        return [
            vf.Assessment(
                assessment_id="guard-finding",
                run_id=run.run_id,
                subject=subject,
                view_id=view.view_id,
                signal=signal,
                status="valid",
                value=0,
            )
        ]

    batch = await execute_assessment(assess, request, complete, [])
    assert batch.run.status == "complete"
    assert batch.assessments[0].value == 0
    assert vf.AssessmentBatch.model_validate_json(batch.model_dump_json()) == batch
    with pytest.raises(ValueError, match="supplied source"):
        await execute_assessment(assess, request, dispatch, [])


async def test_retrospective_context_has_executor_source_independent_of_transformed_view():
    from verifiers.v1.assessment_runtime import execute_assessment

    source = _execution_source()
    subject = _execution_subject(source)
    view = vf.ObservationView.capture({"transformed": "assessor-specific input"},
        snapshot_id=source.snapshot_id, builder_revision="custom", scope="retrospective", subjects=(subject,))
    signal = vf.SignalDefinition(signal_id="anchor", revision="1", semantics="other",
        description="sealed source access", units="indicator")
    run = vf.AssessmentRun(run_id="anchor-run", producer_id="deterministic", producer_revision="1",
        rubric_revision="1", snapshot_id=source.snapshot_id, invocation_id="anchor-call", attempt_id="anchor-attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),))
    request = vf.AssessmentRequest(source=source.identity, run=run, views=(view,))
    with pytest.raises(ValueError, match="outside native execution"):
        vf.AssessmentContext(views=(view,)).retrospective_source()

    def assess(request, context):
        actual = context.retrospective_source()
        assert actual == source and actual is not source
        assert actual.identity == request.source
        assert context.input(view.view_id) == {"transformed": "assessor-specific input"}
        assert "sealed_source" not in context.model_dump_json()
        return (vf.Assessment(assessment_id="anchor-finding", run_id=run.run_id, subject=subject,
            view_id=view.view_id, signal=signal, status="valid", value=1),)

    batch = await execute_assessment(assess, request, source, [])
    assert batch.run.status == "complete" and batch.assessments[0].value == 1


def test_execution_harness_origins_decisions_and_coordinates():
    from verifiers.v1.assessments import canonical_json
    from verifiers.v1.interception.tool import ToolHookRequest
    from verifiers.v1.trace import ToolExecutionEvent

    events = []
    call = vf.ToolCall(id="provider-id", name="send", arguments='{"to":"a"}')
    for index, phase in enumerate(("before", "dispatch", "after")):
        request = ToolHookRequest(
            execution_id="uuid-1",
            event_index=index,
            phase=phase,
            call=call,
            message=vf.ToolMessage(
                tool_call_id=call.id, content="result" if phase == "after" else ""
            ),
        )
        events.append(
            ToolExecutionEvent(
                execution_id="uuid-1",
                event_index=index,
                phase=phase,
                source="interceptor",
                receipt_seq=index,
                node_index=0,
                emitted_call_index=0,
                request_json=canonical_json(request.model_dump(mode="json")),
                decision_json='{"action":"allow"}',
            ).model_dump(mode="json")
        )
    payload = {"trace_id": "trace", "tool_execution_events": events}
    refs = vf.execution_refs(payload, episode_id="episode", trace_id="trace")
    assert refs[0].origin == "interceptor"
    assert refs[0].phase == "after"
    origin_collision = json.loads(canonical_json(payload))
    extra = json.loads(canonical_json(events[0]))
    extra.update(source="harness", receipt_seq=len(events))
    origin_collision["tool_execution_events"].append(extra)
    with pytest.raises(ValueError, match="origin collision is unsupported"):
        vf.execution_refs(origin_collision, episode_id="episode", trace_id="trace")
    for field in ("node_index", "emitted_call_index", "generated_attempt_index"):
        modified = json.loads(canonical_json(payload))
        modified["tool_execution_events"][1][field] = 1
        with pytest.raises(ValueError, match="coordinates"):
            vf.execution_refs(modified, episode_id="episode", trace_id="trace")
    blocked = json.loads(canonical_json(payload))
    blocked["tool_execution_events"][0]["decision_json"] = '{"action":"stop"}'
    with pytest.raises(ValueError, match="nonallowed"):
        vf.execution_refs(blocked, episode_id="episode", trace_id="trace")
    # Actual reporter origin disambiguates equal IDs; arguments/provider IDs do not join them.
    server = json.loads(_execution_source().source_json)["tool_execution_events"]
    for event in server:
        event["receipt_seq"] += len(events)
    payload["tool_execution_events"] += server
    refs = vf.execution_refs(payload, episode_id="episode", trace_id="trace")
    assert len(refs) == 2
    assert refs[0].invocation_id == refs[1].invocation_id
    assert refs[0].occurrence_id != refs[1].occurrence_id


def test_execution_episode_sources_resolve_unique_children():
    from verifiers.v1.episode_assessment import capture_episode_source
    from verifiers.v1.trace import ToolServerExecutionEvent

    traces = []
    raw = json.loads(_execution_source().source_json)
    for trace_id in ("child-1", "child-2"):
        traces.append(
            vf.Trace(
                id=trace_id,
                episode_id="episode",
                agent=vf.AgentInfo(config=vf.AgentConfig()),
                task=vf.TraceTask(type="HookTask", data=HookData(idx=0, prompt="task")),
                tool_execution_events=[
                    ToolServerExecutionEvent.model_validate(event)
                    for event in raw["tool_execution_events"]
                ],
            )
        )
    episode = vf.Episode(id="episode", task=traces[0].task, traces=traces)
    source = capture_episode_source(episode, finalization_state="complete")
    assert len(source.executions) == 2
    left, right = source.executions
    assert left.invocation_id == right.invocation_id
    assert left.occurrence_id != right.occurrence_id
    assert vf.resolve_execution(source, left) == vf.resolve_execution(source, right)
    assert vf.SourceSnapshot.model_validate_json(source.model_dump_json()) == source
    for ref in source.executions:
        subject = _execution_subject(source, ref)
        assert vf.project_subject(subject, source, {}).status == "unsupported"
        vf.capture_execution_view(source, (subject,), scope="through_action_results")
    for duplicate in (False, True):
        broken = json.loads(source.source_json)
        broken["traces"] = [broken["traces"][0]] * (2 if duplicate else 1)
        with pytest.raises(ValueError, match="exactly one retained child"):
            vf.SourceSnapshot.capture(
                broken,
                episode_id="episode",
                trace_ids=source.trace_ids,
                executions=source.executions,
            )


PLUGGED_FNS_PY = """
from verifiers.v1 import Trace


async def reply_length(trace) -> float:
    return float(len(trace.last_reply or ""))


async def exact_match(task, trace) -> float:
    return float(task.answer in (trace.last_reply or ""))


async def marker(trace) -> float:
    return 0.125


async def two_turns(trace: Trace) -> bool:
    return trace.num_turns >= 2
"""


class HookData(vf.TaskData):
    answer: str = ""


class HookTask(vf.Task[HookData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward
    async def lcs(self, trace: vf.Trace) -> float:
        return 0.5

    @vf.reward(weight=0.0)
    async def fmt(self, trace: vf.Trace) -> float:
        return 1.0


async def test_config_plugged_fns_merge_and_override(tmp_path) -> None:
    fns = tmp_path / "fns.py"
    fns.write_text(PLUGGED_FNS_PY)
    config = vf.TaskConfig(
        stops={"single_turn": vf.DecoratedFunctionConfig(fn=f"{fns}:two_turns")},
        metrics={"reply_length": vf.DecoratedFunctionConfig(fn=f"{fns}:reply_length")},
        rewards={
            "exact_match": vf.RewardFunctionConfig(fn=f"{fns}:exact_match", weight=0.5),
            "lcs": vf.RewardFunctionConfig(fn=f"{fns}:marker", weight=2.0),
            "fmt": vf.RewardFunctionConfig(weight=1.0),
        },
    )
    task = HookTask(HookData(idx=0, prompt="abc", answer="cba"), config)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="abc"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="cba"), sampled=True
            ),
        ],
    )

    # the plugged `single_turn` replaces the decorated one: no stop after one turn
    (stop,) = task.hooks("stop")
    assert stop.__name__ == "single_turn"
    assert not await stop(trace)

    await task.score(trace)
    # `lcs` scored by the plugged marker (config wins the name clash), config weight
    assert trace.rewards["lcs"].score == 0.125
    assert trace.rewards["lcs"].weight == 2.0
    assert trace.rewards["exact_match"].score == 1.0
    assert trace.rewards["exact_match"].weight == 0.5
    assert trace.metrics["reply_length"] == 3.0
    # `fmt` keeps the decorated body, only its weight is overridden (0.0 -> 1.0)
    assert trace.rewards["fmt"].score == 1.0
    assert trace.rewards["fmt"].weight == 1.0

    # a fn-less entry must name an existing decorated method
    config = vf.TaskConfig(rewards={"nope": vf.RewardFunctionConfig(weight=1.0)})
    with pytest.raises(ValueError, match="no @vf.reward method named 'nope'"):
        HookTask(HookData(idx=0, prompt="abc"), config).hooks("reward")


async def test_defer_scoring_defers_only_task_scoring() -> None:
    task = HookTask(HookData(idx=0, prompt="abc"))
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HookTask", data=task.data),
    )

    await task.defer_scoring().score(trace)
    assert trace.rewards == {}
    await task.score(trace)
    assert set(trace.rewards) == {"fmt", "lcs"}


def test_compare_stdout_results_accepts_token_equal_text() -> None:
    assert vf.compare_stdout_results("hello   world\n", "hello world\n")


def test_compare_stdout_results_keeps_numeric_tolerance() -> None:
    assert vf.compare_stdout_results("1.0001 2.0\n", "1.0002 2.0009\n")


def test_parse_pytest_outcomes_strips_xfail_xpass_reasons() -> None:
    output = (
        "XFAIL tests/test_mod.py::test_xfail - known bug - still tracked\n"
        "XPASS tests/test_mod.py::test_xpass - always xfail - unexpectedly passed\n"
        "FAILED tests/test_mod.py::test_param[a - b] - assert left - right\n"
        "PASSED tests/test_mod.py::test_ok"
    )

    assert vf.parse_pytest_outcomes(output) == {
        "tests/test_mod.py::test_xfail": "XFAIL",
        "tests/test_mod.py::test_xpass": "XPASS",
        "tests/test_mod.py::test_param[a - b]": "FAILED",
        "tests/test_mod.py::test_ok": "PASSED",
    }


def test_parse_judge_choice_prefers_final_marker_then_boxed() -> None:
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment:\nReasoning mentions B",
            choices=("A", "B"),
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{B}\nFinal Judgment: This is a \\boxed{B}",
            choices=("A", "B"),
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: A better choice is \\boxed{B}", choices=("A", "B")
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice("Draft: \\boxed{A}\nAnswer: B", choices=("A", "B")) == "A"
    )


def _archive_scoring_fixture(copies=8):
    """Small genuine native execution prefixes with repeated assessment journals."""
    from verifiers.v1.assessments import canonical_json
    from verifiers.v1.trace import ToolServerExecutionEvent

    sources = []
    for returned in (False, True):
        source = _execution_source(returned=returned)
        material = json.loads(source.source_json)
        material["archive_padding"] = "ARCHIVE_PAYLOAD" * 2048
        sources.append(
            vf.SourceSnapshot.capture(
                material,
                episode_id=source.episode_id,
                trace_ids=source.trace_ids,
                executions=source.executions,
            )
        )
    dispatch, complete = sources
    signal = vf.SignalDefinition(
        signal_id="archive.harm",
        revision="1",
        semantics="other",
        description="Retained factual finding",
        units="binary",
    )
    batches = []
    for index in range(copies):
        subject = _execution_subject(complete)
        view = vf.capture_execution_view(
            complete, (subject,), scope="prefix", observed=dispatch.executions
        )
        run = vf.AssessmentRun(
            run_id=f"archive-run-{index}",
            producer_id="native-test",
            producer_revision="1",
            rubric_revision="1",
            snapshot_id=complete.snapshot_id,
            invocation_id=f"archive-invocation-{index}",
            attempt_id=f"archive-attempt-{index}",
            expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
        )
        assessment = vf.Assessment(
            assessment_id=f"archive-finding-{index}",
            run_id=run.run_id,
            subject=subject,
            view_id=view.view_id,
            signal=signal,
            status="valid",
            value=1,
        )
        batches.append(
            vf.AssessmentBatch(
                source=complete, run=run, views=(view,), assessments=(assessment,)
            )
        )
    subject = _execution_subject(dispatch)
    view = vf.capture_execution_view(
        dispatch, (subject,), scope="prefix", observed=dispatch.executions
    )
    for index, status in enumerate(("queued", "running", "failed", "interrupted")):
        run = vf.AssessmentRun(
            run_id=f"archive-history-{index}",
            producer_id="native-test",
            producer_revision="1",
            rubric_revision="1",
            snapshot_id=dispatch.snapshot_id,
            invocation_id=f"archive-history-invocation-{index}",
            attempt_id=f"archive-history-attempt-{index}",
            expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
            status=status,
        )
        batches.append(vf.AssessmentBatch(source=dispatch, run=run, views=(view,)))
    parent = batches[0].assessments[0]
    request = vf.CreditRequest(
        source=complete,
        invocation_id="archive-credit",
        attempt_id="archive-credit-attempt",
        rule=vf.CreditRule(rule_id="archive-test", revision="1"),
        accepted=(parent,),
        targets=(vf.CreditTarget(recipient=parent.subject),),
        allocation="fixed_mass",
        overlap_policy="reject",
    )
    contribution = vf.CreditContribution(
        contribution_id="archive-contribution",
        parent_assessment_ids=(parent.assessment_id,),
        recipient=parent.subject,
        signal=signal,
        transformation="identity",
        status="valid",
        value=1,
        allocation="fixed_mass",
        attribution="coarse",
    )
    assignment = vf.CreditAssignment(request=request, contributions=(contribution,))
    trace = vf.Trace(
        id="trace",
        episode_id="episode",
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="send")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        tool_execution_events=tuple(
            ToolServerExecutionEvent.model_validate(event)
            for event in json.loads(complete.source_json)["tool_execution_events"]
        ),
        assessment_batches=batches,
        credit_assignments=[assignment],
        is_completed=True,
        ok=True,
    )
    # Python-mode remains inline and can represent old archives independently.
    legacy = trace.model_dump(mode="python")
    assert "source_json" in legacy["assessment_batches"][0]["source"]
    assert canonical_json(json.loads(complete.source_json)) == complete.source_json
    return trace, legacy


def test_scoring_archive_sources_are_once_per_owner_and_roundtrip_prefix_history():
    trace, legacy = _archive_scoring_fixture()
    payload = json.loads(trace.model_dump_json())
    assert len(payload["assessment_sources"]) == 2
    assert all(
        "source_json" not in batch["source"] for batch in payload["assessment_batches"]
    )
    assert "source_json" not in payload["credit_assignments"][0]["request"]["source"]
    restored = vf.WireTrace.model_validate_json(json.dumps(payload))
    assert restored.assessment_batches == trace.assessment_batches
    assert restored.credit_assignments == trace.credit_assignments
    assert [batch.run.status for batch in restored.assessment_batches] == [
        batch.run.status for batch in trace.assessment_batches
    ]
    assert all(
        isinstance(batch.source, vf.SourceSnapshot)
        for batch in restored.assessment_batches
    )
    first = restored.assessment_batches[0]
    assert first.source is restored.assessment_batches[1].source
    assert first.source is restored.credit_assignments[0].request.source
    observed = json.loads(first.views[0].input_json)[0]
    assert len(observed["events"]) == 1
    assert json.loads(observed["events"][0]["receipt_json"])["result_json"] is None
    assert (
        vf.resolve_execution(first.source, first.source.executions[0])[-1]["phase"]
        == "returned"
    )
    assert (
        vf.WireTrace.model_validate(legacy).assessment_batches
        == trace.assessment_batches
    )
    assert "source_json" in json.loads(first.model_dump_json())["source"]
    assert (
        "source_json"
        in json.loads(restored.credit_assignments[0].model_dump_json())["request"][
            "source"
        ]
    )
    episode = vf.WireEpisode(
        id="episode",
        task=trace.task.model_dump(mode="json"),
        traces=[restored],
        assessment_batches=list(trace.assessment_batches),
        credit_assignments=list(trace.credit_assignments),
    )
    episode_payload = json.loads(episode.model_dump_json())
    assert len(episode_payload["assessment_sources"]) == 2
    assert len(episode_payload["traces"][0]["assessment_sources"]) == 2
    reloaded = vf.WireEpisode.model_validate_json(json.dumps(episode_payload))
    assert reloaded.assessment_batches == episode.assessment_batches
    assert reloaded.credit_assignments == episode.credit_assignments
    assert reloaded.traces[0].assessment_batches == restored.assessment_batches
    missing_owner = copy.deepcopy(episode_payload)
    missing_owner.pop("assessment_sources")
    with pytest.raises(ValueError):
        vf.WireEpisode.model_validate_json(json.dumps(missing_owner))


def test_scoring_archive_size_grows_with_journal_metadata_not_source_copies():
    trace, inline = _archive_scoring_fixture(copies=12)
    compact = trace.model_dump_json()
    legacy = json.dumps(inline)
    assert compact.count("ARCHIVE_PAYLOAD" * 2048) == 2
    assert len(compact) < len(legacy) / 2
    larger, _ = _archive_scoring_fixture(copies=24)
    assert len(larger.model_dump_json()) - len(compact) < 12 * 6000


@pytest.mark.parametrize(
    "mutation", ["dangling", "wrong-identity", "duplicate", "unused", "changed-json"]
)
def test_scoring_archive_rejects_invalid_source_pool(mutation):
    trace, _ = _archive_scoring_fixture(copies=1)
    payload = json.loads(trace.model_dump_json())
    if mutation == "dangling":
        payload["assessment_sources"] = []
    elif mutation == "wrong-identity":
        payload["assessment_batches"][0]["source"]["episode_id"] = "another-episode"
    elif mutation == "duplicate":
        payload["assessment_sources"].append(
            copy.deepcopy(payload["assessment_sources"][0])
        )
    elif mutation == "unused":
        unused = vf.SourceSnapshot.capture(
            {}, episode_id="episode", trace_ids=("trace",)
        )
        payload["assessment_sources"].append(unused.model_dump(mode="json"))
    else:
        payload["assessment_sources"][0]["source_json"] = "{}"
    with pytest.raises(ValueError):
        vf.WireTrace.model_validate_json(json.dumps(payload))


def test_scoring_archive_rejects_model_copy_source_tampering():
    trace, _ = _archive_scoring_fixture(copies=1)
    batch = trace.assessment_batches[0]
    bad_source = batch.source.model_copy(update={"source_json": "{}"})
    bad_batch = batch.model_copy(update={"source": bad_source})
    bad_trace = trace.model_copy(update={"assessment_batches": [bad_batch]})
    with pytest.raises(ValueError):
        bad_trace.model_dump_json()


@pytest.mark.parametrize("where", ["reference", "pooled-source"])
def test_scoring_archive_rejects_boolean_node_identity_forgery(where):
    from verifiers.v1.assessment_archive import normalize_history, restore_history

    source = vf.SourceSnapshot.capture(
        {},
        episode_id="episode",
        trace_ids=("trace",),
        nodes=(
            vf.NodeRef(
                trace_id="trace", node_index=1, node_content_digest="node-digest"
            ),
        ),
    )
    archived = normalize_history({"assessment_batches": [{"source": source}]})
    if where == "reference":
        archived["assessment_batches"][0]["source"]["nodes"][0]["node_index"] = True
    else:
        archived["assessment_sources"][0]["nodes"][0]["node_index"] = True
    with pytest.raises(ValueError):
        restore_history(archived)


def test_scoring_archive_pool_resolution_preserves_native_prefix_verification():
    from verifiers.v1.assessments import canonical_json

    trace, _ = _archive_scoring_fixture(copies=1)
    payload = json.loads(trace.model_dump_json())
    source = _execution_source(returned=False)
    foreign = json.loads(source.source_json)
    envelope = json.loads(foreign["tool_execution_events"][0]["receipt_json"])
    envelope["arguments_json"] = '{"to":"rewritten"}'
    foreign["tool_execution_events"][0]["receipt_json"] = canonical_json(envelope)
    refs = vf.execution_refs(foreign, episode_id="episode", trace_id="trace")
    changed = vf.SourceSnapshot.capture(
        foreign, episode_id="episode", trace_ids=("trace",), executions=refs
    )
    batch = trace.assessment_batches[0]
    forged_view = vf.ObservationView.capture(
        [
            {
                "subject_id": batch.views[0].subjects[0].subject_id,
                "observed": refs[0].model_dump(mode="json"),
                "events": list(vf.resolve_execution(changed, refs[0])),
            }
        ],
        snapshot_id=batch.source.snapshot_id,
        builder_revision="native_execution_prefix_v1",
        scope="prefix",
        subjects=batch.views[0].subjects,
    )
    payload["assessment_batches"][0]["views"] = [forged_view.model_dump(mode="json")]
    payload["assessment_batches"][0]["assessments"][0]["view_id"] = forged_view.view_id
    payload["assessment_views"] = [
        view
        for view in payload["assessment_views"]
        if view["view_id"] != batch.views[0].view_id
    ]
    with pytest.raises(ValueError, match="prefix"):
        vf.WireTrace.model_validate_json(json.dumps(payload))


def test_scoring_archive_normalization_preserves_input_and_source_string_sharing():
    from verifiers.v1.assessment_archive import normalize_history

    trace, legacy = _archive_scoring_fixture(copies=1)
    unchanged = copy.deepcopy(legacy)
    compact = normalize_history(legacy)
    assert legacy == unchanged
    assert (
        compact["assessment_sources"][0]["source_json"]
        is legacy["assessment_batches"][0]["source"]["source_json"]
    )
    assert "assessment_sources" not in normalize_history(
        {"assessment_batches": [], "credit_assignments": []}
    )
    assert trace.assessment_batches[0].source.source_json


def _view_archive_scoring_fixture(copies=8):
    trace, _ = _archive_scoring_fixture(copies=copies)
    first = trace.assessment_batches[0]
    body = {"prepared_context": "VIEW_PAYLOAD" * 4096}
    views = tuple(
        vf.ObservationView.capture(
            body,
            snapshot_id=first.source.snapshot_id,
            builder_revision="arbitrary-preparation-v1",
            scope=scope,
            subjects=first.views[0].subjects,
        )
        for scope in ("retrospective", "through_action_results")
    )
    batches = []
    for batch in trace.assessment_batches[:copies]:
        raw = batch.model_dump(mode="python")
        raw["views"] = views
        raw["assessments"][0]["view_id"] = views[0].view_id
        batches.append(vf.AssessmentBatch.model_validate(raw))
    trace = trace.model_copy(
        update={"assessment_batches": batches + trace.assessment_batches[copies:]}
    )
    return trace, trace.model_dump(mode="python")


@pytest.mark.parametrize("load_api", ["json", "python", "adapter"])
def test_scoring_archive_views_are_once_per_identity_with_exact_context(monkeypatch, load_api):
    from pydantic import TypeAdapter

    from verifiers.v1 import assessments
    from verifiers.v1._validation_scope import _owner, validation_scope
    from verifiers.v1.assessment_archive import normalize_history

    trace, legacy = _view_archive_scoring_fixture()
    unchanged = copy.deepcopy(legacy)
    normalized = normalize_history(legacy)
    assert legacy == unchanged
    assert (
        normalized["assessment_views"][0]["input_json"]
        is legacy["assessment_batches"][0]["views"][0]["input_json"]
    )
    payload = json.loads(trace.model_dump_json())
    large = [
        view
        for view in payload["assessment_views"]
        if "VIEW_PAYLOAD" in view["input_json"]
    ]
    assert len(large) == 2
    assert large[0]["input_digest"] == large[1]["input_digest"]
    assert large[0]["view_id"] != large[1]["view_id"]
    reference = payload["assessment_batches"][0]["views"][0]
    assert set(reference) == {"archive_view_ref"}
    assert "input_json" not in reference["archive_view_ref"]
    calls = []
    original = assessments._canonical

    def observed(text):
        calls.append(text)
        return original(text)

    def load(model, data):
        if load_api == "json":
            return model.model_validate_json(json.dumps(data))
        if load_api == "adapter":
            return TypeAdapter(model).validate_python(data)
        return model.model_validate(data)

    monkeypatch.setattr(assessments, "_canonical", observed)
    restored = load(vf.WireTrace, payload)
    assert _owner.get() is None
    assert calls.count(large[0]["input_json"]) == 2  # Two distinct typed views.
    for source in payload["assessment_sources"]:
        assert calls.count(source["source_json"]) == 1
    assert restored.assessment_batches == trace.assessment_batches
    assert (
        restored.assessment_batches[0].views[0]
        is restored.assessment_batches[1].views[0]
    )
    assert (
        vf.WireTrace.model_validate(legacy).assessment_batches
        == trace.assessment_batches
    )
    first = restored.assessment_batches[0]
    assert "input_json" in json.loads(first.model_dump_json())["views"][0]
    assert "archive_view_ref" not in json.loads(first.views[0].model_dump_json())
    assert "input_json" in first.views[0].model_dump(mode="python")
    episode = vf.WireEpisode(
        id="episode",
        task=trace.task.model_dump(mode="json"),
        traces=[restored],
        assessment_batches=list(trace.assessment_batches),
        credit_assignments=list(trace.credit_assignments),
    )
    episode_payload = json.loads(episode.model_dump_json())
    assert len(episode_payload["assessment_views"]) == len(payload["assessment_views"])
    assert len(episode_payload["traces"][0]["assessment_views"]) == len(
        payload["assessment_views"]
    )
    calls.clear()
    reloaded = load(vf.WireEpisode, episode_payload)
    assert reloaded.assessment_batches == episode.assessment_batches
    assert reloaded.credit_assignments == episode.credit_assignments
    assert reloaded.traces[0].credit_assignments == trace.credit_assignments
    assert _owner.get() is None
    for source in episode_payload["assessment_sources"]:
        assert calls.count(source["source_json"]) == 1
    assert calls.count(large[0]["input_json"]) == 2
    with validation_scope():
        owner = _owner.get()
        load(vf.WireEpisode, episode_payload)
        assert _owner.get() is owner and not owner.closed
    episode_payload.pop("assessment_views")
    with pytest.raises(ValueError):
        load(vf.WireEpisode, episode_payload)
    assert _owner.get() is None


def test_scoring_archive_view_body_storage_is_bounded_by_unique_views():
    trace, inline = _view_archive_scoring_fixture(copies=12)
    compact = trace.model_dump_json()
    assert compact.count("VIEW_PAYLOAD" * 4096) == 2
    assert len(compact) < len(json.dumps(inline)) / 3
    larger, _ = _view_archive_scoring_fixture(copies=24)
    assert len(larger.model_dump_json()) - len(compact) < 12 * 6000


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unused",
        "duplicate",
        "conflicting",
        "metadata",
        "unknown-reference-field",
    ],
)
def test_scoring_archive_rejects_invalid_view_pool(mutation):
    trace, _ = _view_archive_scoring_fixture(copies=1)
    payload = json.loads(trace.model_dump_json())
    pool = payload["assessment_views"]
    if mutation == "missing":
        payload.pop("assessment_views")
    elif mutation == "unused":
        first = trace.assessment_batches[0]
        unused = vf.ObservationView.capture(
            "unused context",
            snapshot_id=first.source.snapshot_id,
            builder_revision="unused-v1",
            scope="retrospective",
            subjects=first.views[0].subjects,
        )
        pool.append(unused.model_dump(mode="json"))
    elif mutation == "duplicate":
        pool.append(copy.deepcopy(pool[0]))
    elif mutation == "conflicting":
        conflicting = copy.deepcopy(pool[0])
        conflicting["input_json"] = '"changed context"'
        payload["assessment_batches"][0]["views"][0] = conflicting
    elif mutation == "metadata":
        payload["assessment_batches"][0]["views"][0]["archive_view_ref"]["scope"] = (
            "prefix"
        )
    else:
        payload["assessment_batches"][0]["views"][0]["archive_view_ref"]["unknown"] = (
            True
        )
    with pytest.raises(ValueError):
        vf.WireTrace.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("where", ["pooled-view", "reference"])
@pytest.mark.parametrize("forged_index", [True, "1"])
def test_scoring_archive_view_metadata_rejects_coerced_node_index(where, forged_index):
    from verifiers.v1.assessment_archive import normalize_history, restore_history

    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    subject = vf.SubjectRef(
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        kind="turn",
        trace_id="trace",
        node_index=1,
        node_content_digest="node-digest",
    )
    view = vf.ObservationView.capture(
        "context",
        snapshot_id=source.snapshot_id,
        builder_revision="generic-v1",
        scope="retrospective",
        subjects=(subject,),
    )
    payload = normalize_history(
        {"assessment_batches": [{"source": source, "views": [view]}]}
    )
    target = (
        payload["assessment_views"][0]
        if where == "pooled-view"
        else payload["assessment_batches"][0]["views"][0]["archive_view_ref"]
    )
    target["subjects"][0]["node_index"] = forged_index
    with pytest.raises(ValueError):
        restore_history(payload)


def test_scoring_archive_rejects_model_copy_view_tampering():
    trace, _ = _view_archive_scoring_fixture(copies=1)
    batch = trace.assessment_batches[0]
    bad_view = batch.views[0].model_copy(update={"input_json": '"tampered"'})
    bad_batch = batch.model_copy(update={"views": (bad_view, batch.views[1])})
    bad_trace = trace.model_copy(update={"assessment_batches": [bad_batch]})
    with pytest.raises(ValueError):
        bad_trace.model_dump_json()


def test_scoring_archive_artifact_views_remain_inline_and_legacy_compatible():
    from verifiers.v1.assessments import content_digest

    trace, _ = _view_archive_scoring_fixture(copies=1)
    batch = trace.assessment_batches[0]
    raw = batch.views[0].model_dump(mode="json", exclude={"view_id"})
    raw["input_json"] = None
    raw["artifact"] = {
        "uri": "artifact://prepared-context",
        "digest": raw["input_digest"],
        "media_type": "application/json",
    }
    artifact_view = vf.ObservationView.model_validate(
        dict(raw, view_id=content_digest(raw))
    )
    batch_raw = batch.model_dump(mode="python")
    batch_raw["views"] = (artifact_view,)
    batch_raw["assessments"][0]["view_id"] = artifact_view.view_id
    artifact_batch = vf.AssessmentBatch.model_validate(batch_raw)
    trace = trace.model_copy(
        update={"assessment_batches": [artifact_batch] + trace.assessment_batches[1:]}
    )
    payload = json.loads(trace.model_dump_json())
    assert (
        payload["assessment_batches"][0]["views"][0]["artifact"]["uri"]
        == "artifact://prepared-context"
    )
    assert artifact_view.view_id not in {
        view["view_id"] for view in payload["assessment_views"]
    }
    assert vf.WireTrace.model_validate_json(json.dumps(payload)).assessment_batches[
        0
    ].views == (artifact_view,)
    # Old artifact archives may omit their null input_json field.
    payload["assessment_batches"][0]["views"][0].pop("input_json")
    assert vf.WireTrace.model_validate_json(json.dumps(payload)).assessment_batches[
        0
    ].views == (artifact_view,)


def test_scoring_archive_view_reference_cannot_cross_batch_snapshot():
    trace, _ = _archive_scoring_fixture(copies=1)
    payload = json.loads(trace.model_dump_json())
    first = payload["assessment_batches"][0]
    foreign_view = trace.assessment_batches[1].views[0]
    first["views"] = copy.deepcopy(payload["assessment_batches"][1]["views"])
    first["assessments"][0]["view_id"] = foreign_view.view_id
    payload["assessment_views"] = [
        view
        for view in payload["assessment_views"]
        if view["view_id"] != trace.assessment_batches[0].views[0].view_id
    ]
    with pytest.raises(ValueError, match="view/source"):
        vf.WireTrace.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_source_snapshot_copied_schema_requires_exact_integer(schema_version):
    source = vf.SourceSnapshot.capture({}, episode_id="episode")
    copied = source.model_copy(update={"schema_version": schema_version})
    with pytest.raises(ValueError, match="exact integer"):
        vf.SourceSnapshot.model_validate(copied)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", False, 2.0])
def test_source_snapshot_raw_schema_requires_exact_integer(schema_version):
    source = vf.SourceSnapshot.capture({}, episode_id="episode")
    raw = source.model_dump(mode="python")
    raw["schema_version"] = schema_version
    with pytest.raises(ValueError, match="exact integer"):
        vf.SourceSnapshot.model_validate(raw)


@pytest.mark.parametrize("with_execution", [False, True])
def test_source_snapshot_integer_schema_preserves_native_and_legacy_roundtrips(
    with_execution,
):
    source = (
        _execution_source(returned=True)
        if with_execution
        else vf.SourceSnapshot.capture({}, episode_id="episode")
    )
    assert type(source.schema_version) is int
    assert source.schema_version == (2 if with_execution else 1)
    assert vf.SourceSnapshot.model_validate_json(source.model_dump_json()) == source
    legacy = source.model_dump(mode="python")
    if not with_execution:
        legacy.pop("schema_version")
    assert vf.SourceSnapshot.model_validate(legacy) == source


@pytest.mark.parametrize("coordinate", [True, 1.0, "1"])
@pytest.mark.parametrize("warm", [False, True])
def test_intrinsic_source_proofs_reject_raw_and_copied_coordinate_coercion(coordinate, warm):
    from verifiers.v1._validation_scope import validation_scope

    node = vf.NodeRef(trace_id="trace", node_index=1, node_content_digest="node")
    source = vf.SourceSnapshot.capture({}, episode_id="episode", nodes=(node,), trace_ids=("trace",))
    with validation_scope():
        if warm:
            source.verify()
        copied = source.model_copy(update={"nodes": (node.model_copy(update={"node_index": coordinate}),)})
        with pytest.raises(ValueError):
            copied.verify()
        raw = source.model_dump(mode="python")
        raw["nodes"][0]["node_index"] = coordinate
        with pytest.raises(ValueError):
            vf.SourceSnapshot.model_validate(raw)


@pytest.mark.parametrize("coordinate", [True, 1.0, "1"])
@pytest.mark.parametrize("field", ["node_index", "call_index", "span_start", "span_end"])
def test_intrinsic_view_proofs_reject_nested_group_coordinate_coercion(coordinate, field):
    from verifiers.v1._validation_scope import validation_scope

    kind = "call" if field == "call_index" else "span" if field.startswith("span") else "turn"
    extra = {"call_index": 1} if kind == "call" else {
        "span_start": 1, "span_end": 2, "representation": "full_tokens", "representation_digest": "tokens"} if kind == "span" else {}
    subject = vf.SubjectRef(kind=kind, snapshot_id="snapshot", episode_id="episode", trace_id="trace",
                            node_index=1, node_content_digest="node", **extra)
    group = vf.SubjectRef(kind="group", snapshot_id="snapshot", episode_id="episode", members=(subject,))
    view = vf.ObservationView.capture({}, snapshot_id="snapshot", builder_revision="typed", scope="retrospective", subjects=(group,))
    with validation_scope():
        view.verify()
        copied = view.model_copy(update={"subjects": (group.model_copy(update={"members": (
            subject.model_copy(update={field: coordinate}),)}),)})
        with pytest.raises(ValueError):
            copied.verify()
        raw = view.model_dump(mode="python")
        raw["subjects"][0]["members"][0][field] = coordinate
        with pytest.raises(ValueError):
            vf.ObservationView.model_validate(raw)


@pytest.mark.parametrize("count", [True, 2.0, "2"])
def test_intrinsic_source_proof_rejects_copied_execution_count(count):
    from verifiers.v1._validation_scope import validation_scope

    source = _execution_source()
    with validation_scope():
        source.verify()
        copied = source.model_copy(update={"executions": (source.executions[0].model_copy(update={"event_count": count}),)})
        with pytest.raises(ValueError):
            copied.verify()


def test_intrinsic_proof_hits_require_exact_body_metadata_and_success(monkeypatch):
    from verifiers.v1 import assessments
    from verifiers.v1._validation_scope import _owner, validation_scope

    source = vf.SourceSnapshot.capture({"body": "original"}, episode_id="episode")
    calls = []
    original = assessments._canonical
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(assessments, "_canonical", observed)
    with validation_scope():
        source.verify()
        source.verify()
        assert calls.count(source.source_json) == 1
        owner = _owner.get()
        for changes in ({"source_json": '{"body":"changed"}'}, {"source_digest": "changed"},
                        {"episode_id": "other"}, {"snapshot_id": "other"}):
            with pytest.raises(ValueError):
                source.model_copy(update=changes).verify()
        assert len(owner.proofs) == 1
    assert owner.closed and not owner.proofs and owner.retained_bytes == 0
    source.verify()
    assert calls.count(source.source_json) > 1


@pytest.mark.parametrize("limit", ["entries", "bytes"])
def test_intrinsic_proof_eviction_and_oversize_keep_full_validation(monkeypatch, limit):
    from verifiers.v1 import _validation_scope as scopes
    from verifiers.v1 import assessments

    sources = [vf.SourceSnapshot.capture({"body": str(index)}, episode_id="episode") for index in range(2)]
    calls = []
    original = assessments._canonical
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(assessments, "_canonical", observed)
    monkeypatch.setattr(scopes, "_MAX_ENTRIES" if limit == "entries" else "_MAX_BYTES", 1)
    with scopes.validation_scope():
        for source in (*sources, sources[0]):
            source.verify()
        assert calls.count(sources[0].source_json) == 2
        assert scopes._owner.get().retained_bytes <= scopes._MAX_BYTES


def test_compact_large_unicode_proofs_reuse_without_normalizing_content(monkeypatch):
    from verifiers.v1 import _validation_scope as scopes
    from verifiers.v1 import assessments

    source = vf.SourceSnapshot.capture({"body": "source" * 1000 + "😀"}, episode_id="episode")
    subject = vf.SubjectRef(kind="episode", snapshot_id=source.snapshot_id, episode_id="episode")
    view = vf.ObservationView.capture({"body": "view" * 2000 + "😀"}, snapshot_id=source.snapshot_id,
        builder_revision="test", scope="retrospective", subjects=(subject,))
    calls = []
    original = assessments._canonical
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(assessments, "_canonical", observed)
    with scopes.validation_scope():
        sizes = [scopes._size(scopes.intrinsic_proof(model).key) + 256 for model in (source, view)]
        monkeypatch.setattr(scopes, "_MAX_BYTES", sum(sizes))
        for _ in range(3):
            source.verify()
            view.verify()
        assert calls.count(source.source_json) == 1
        assert calls.count(view.input_json) == 1
        assert scopes.intrinsic_proof(source).hit
        assert scopes.intrinsic_proof(view).hit
        assert scopes._owner.get().retained_bytes <= scopes._MAX_BYTES
        with pytest.raises(ValueError):
            source.model_copy(update={"source_digest": "tampered"}).verify()
        with pytest.raises(ValueError):
            view.model_copy(update={"input_digest": "tampered"}).verify()
        assert scopes._typed("x" * 5000 + "é") != scopes._typed("x" * 5000 + "e\u0301")
        surrogate = "x" * 5000 + "\ud800"
        assert scopes._typed(surrogate) == (str, surrogate)
        assert scopes._typed(surrogate)[1] is surrogate


async def test_intrinsic_executor_children_reuse_but_unrelated_plans_and_reload_are_isolated(monkeypatch):
    from verifiers.v1 import assessments
    from verifiers.v1.assessment_runtime import execute_assessment_plan

    source = _execution_source()
    subject = _execution_subject(source)
    view = vf.ObservationView.capture({"working": "input"}, snapshot_id=source.snapshot_id,
        builder_revision="private-proof-test", scope="retrospective", subjects=(subject,))
    signal = vf.SignalDefinition(signal_id="proof", revision="1", semantics="other", description="proof", units="binary")
    def requests(prefix):
        return [("assess", vf.AssessmentRequest(source=source.identity, views=(view,), run=vf.AssessmentRun(
            run_id=f"{prefix}-run-{index}", producer_id="deterministic", producer_revision="1", rubric_revision="1",
            snapshot_id=source.snapshot_id, invocation_id=f"{prefix}-call-{index}", attempt_id=f"{prefix}-attempt-{index}",
            expected=(vf.AssessmentTarget(subject=subject, signal=signal),)))) for index in range(2)]
    def assess(request, context):
        assert context.retrospective_source() == source
        return (vf.Assessment(assessment_id=request.run.run_id, run_id=request.run.run_id, subject=subject,
            view_id=view.view_id, signal=signal, status="valid", value=1),)
    plans = [requests("a"), requests("b")]
    calls = []
    original = assessments._canonical
    def observed(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(assessments, "_canonical", observed)
    left, right = [], []
    await asyncio.gather(execute_assessment_plan({"assess": assess}, plans[0], source, left, 2),
                         execute_assessment_plan({"assess": assess}, plans[1], source, right, 2))
    assert calls.count(source.source_json) == 2
    assert calls.count(view.input_json) == 2
    assert all(batch.run.status == "complete" for batch in (*left, *right) if batch.run.status in {"complete", "failed"})
    before = calls.count(source.source_json)
    vf.AssessmentBatch.model_validate_json(left[-1].model_dump_json())
    assert calls.count(source.source_json) > before


async def test_intrinsic_owner_rejects_sibling_inheritance_and_escaped_child_after_cancel():
    from verifiers.v1._validation_scope import (
        _owner,
        intrinsic_proof,
        validation_owner,
        validation_scope,
    )

    source = vf.SourceSnapshot.capture({}, episode_id="episode")
    started, finish = asyncio.Event(), asyncio.Event()
    escaped = []
    owners = []
    async def child():
        await finish.wait()
        assert not intrinsic_proof(source).hit
        with validation_scope(borrow=True):
            assert _owner.get() is not owners[0]
            source.verify()
    @validation_owner()
    async def parent():
        source.verify()
        owners.append(_owner.get())
        async def sibling():
            assert not intrinsic_proof(source).hit
            with validation_scope(borrow=True):
                assert _owner.get() is not owners[0]
        await asyncio.create_task(sibling())
        escaped.append(asyncio.create_task(child()))
        started.set()
        await asyncio.Event().wait()
    pending = asyncio.create_task(parent())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert owners[0].closed and not owners[0].proofs
    finish.set()
    await escaped[0]
    assert _owner.get() is None


async def test_intrinsic_cached_source_cannot_launder_copied_producer_coordinate():
    from verifiers.v1.assessment_runtime import execute_assessment

    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",),
        nodes=(vf.NodeRef(trace_id="trace", node_index=1, node_content_digest="node"),))
    subject = vf.SubjectRef(kind="call", snapshot_id=source.snapshot_id, episode_id="episode", trace_id="trace",
        node_index=1, node_content_digest="node", call_index=1)
    view = vf.ObservationView.capture({}, snapshot_id=source.snapshot_id, builder_revision="typed-producer",
        scope="retrospective", subjects=(subject,))
    signal = vf.SignalDefinition(signal_id="typed", revision="1", semantics="other", description="typed", units="binary")
    run = vf.AssessmentRun(run_id="typed-run", producer_id="typed", producer_revision="1", rubric_revision="1",
        snapshot_id=source.snapshot_id, invocation_id="typed-call", attempt_id="typed-attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),))
    request = vf.AssessmentRequest(source=source.identity, run=run, views=(view,))
    def malformed(request, context):
        context.retrospective_source()
        parent = vf.Assessment(assessment_id="typed-result", run_id=run.run_id, subject=subject,
            view_id=view.view_id, signal=signal, status="valid", value=1)
        return (parent.model_copy(update={"subject": subject.model_copy(update={"call_index": True})}),)
    batch = await execute_assessment(malformed, request, source, [])
    assert batch.run.status == "failed" and not batch.assessments


@pytest.mark.parametrize("api", ["task", "env"])
async def test_intrinsic_task_and_episode_scores_have_fresh_owners(api):
    from verifiers.v1._validation_scope import _owner, validation_scope

    observed = []
    class CreditOnlyTask(HookTask):
        def assessment_source(self, trace):
            observed.append(_owner.get())
            return {"public": "task"}

        def credit_requests(self, source, assessments):
            assert _owner.get() is observed[-1]
            return []

        @vf.credit
        async def unused(self, request):
            return ()

    class CreditOnlyEnv(vf.Env):
        async def run(self, task, agents):
            raise NotImplementedError

        def assessment_source(self, task, episode):
            observed.append(_owner.get())
            return {"public": "episode"}

        def credit_requests(self, source, assessments):
            assert _owner.get() is observed[-1]
            return []

        @vf.credit
        async def unused(self, request):
            return ()

    task = CreditOnlyTask(HookData(idx=0, prompt="task"))
    trace = vf.Trace(episode_id="owner-test", task=vf.TraceTask(type="CreditOnlyTask", data=task.data), agent=vf.AgentInfo(config=vf.AgentConfig()))
    with validation_scope():
        outer = _owner.get()
        for _ in range(2):
            if api == "task":
                await task.score_assessments(trace)
            else:
                episode = vf.Episode(id="owner-test", task=trace.task, traces=[trace])
                env = object.__new__(CreditOnlyEnv)
                await env.score_assessments(task, episode, finalization_state="complete")
                assert not episode.assessment_errors and not episode.credit_errors
            assert _owner.get() is outer
            assert observed[-1] is not outer and observed[-1].closed and not observed[-1].proofs
        assert observed[0] is not observed[1]


def test_boxed_math_answer_scores_off_the_main_thread() -> None:
    """math-verify's timeout uses signal.alarm, which only the main thread may
    set; trainers such as veRL score inside Ray actors off the main thread."""
    import threading

    from verifiers.v1.utils.score import verify_boxed_math_answer

    results: dict[str, float] = {}

    def score() -> None:
        results["correct"] = verify_boxed_math_answer("so \\boxed{12}", "12")
        results["wrong"] = verify_boxed_math_answer("so \\boxed{13}", "12")

    worker = threading.Thread(target=score)
    worker.start()
    worker.join()

    assert results == {"correct": 1.0, "wrong": 0.0}
    assert verify_boxed_math_answer("so \\boxed{12}", "12") == 1.0


def test_archive_serialization_elides_repeated_evidence_objects(monkeypatch):
    import importlib

    from pydantic import TypeAdapter

    from verifiers.v1 import assessment_archive, assessments

    credit = importlib.import_module("verifiers.v1.credit")
    trace, _ = _view_archive_scoring_fixture()
    batches = trace.assessment_batches
    assert batches[0].views[0] is batches[1].views[0]
    emitted = []
    original_scope = assessment_archive.archive_serialization

    class Recorder:
        def __init__(self):
            self.inner = original_scope()

        def __enter__(self):
            self.scope = self.inner.__enter__()
            return self.scope

        def __exit__(self, *exc):
            emitted.append(self.scope.emitted)
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(assessment_archive, "archive_serialization", Recorder)
    elided = json.dumps(trace.to_record(), sort_keys=True)
    assert emitted and emitted[-1] > 0
    # Byte-identical to serializing every batch's evidence in full.
    full = lambda value, handler, info: handler(value)
    monkeypatch.setattr(assessments, "_archive_field", full)
    monkeypatch.setattr(credit, "_archive_field", full)
    assert json.dumps(trace.to_record(), sort_keys=True) == elided
    monkeypatch.undo()
    # Only the same object within one scope is elided; field filters are not.
    pair = TypeAdapter(tuple[vf.AssessmentBatch, vf.AssessmentBatch])
    with assessments.archive_serialization() as scope:
        first, second = pair.dump_python((batches[0], batches[0]), mode="json")
        assert "source_json" in first["source"]
        assert second["source"] == {
            assessments.ARCHIVE_SAME: [scope.token, "source", batches[0].source.snapshot_id]
        }
        filtered = pair.dump_python(
            (batches[0], batches[0]), mode="json", exclude={1: {"source": {"nodes"}}}
        )
        assert "source_json" in filtered[1]["source"]
        assert "nodes" not in filtered[1]["source"]
    plain = pair.dump_python((batches[0], batches[0]), mode="json")
    assert plain[0] == plain[1] and "source_json" in plain[1]["source"]
    # A placeholder from another scope (or forged input) is never honored.
    raw = trace.model_dump(mode="python")
    raw["assessment_batches"][1]["source"] = {
        assessments.ARCHIVE_SAME: ["forged", "source", batches[0].source.snapshot_id]
    }
    with pytest.raises(ValueError):
        assessment_archive.normalize_history(raw)


def test_archive_serialization_still_rejects_conflicting_equal_identity_copies():
    from pydantic_core import PydanticSerializationError

    trace, _ = _view_archive_scoring_fixture()
    batches = list(trace.assessment_batches)
    forged = batches[1].source.model_copy(update={"source_json": '{"forged":true}'})
    batches[1] = batches[1].model_copy(update={"source": forged})
    trace = trace.model_copy(update={"assessment_batches": batches})
    with pytest.raises((PydanticSerializationError, ValueError)):
        trace.to_record()


def test_env_server_reply_pools_assessment_evidence_and_restores_identically():
    from verifiers.v1.episode import Episode
    from verifiers.v1.serve.client import _decode_response
    from verifiers.v1.serve.server import _pack_response
    from verifiers.v1.serve.types import RunResponse

    trace, _ = _view_archive_scoring_fixture()
    episode = Episode(id="episode", task=trace.task, traces=[trace])
    # The server's reply carries the env's typed episode, serialized as itself.
    data = _pack_response(
        RunResponse.model_construct(success=True, error=None, episode=episode)
    )
    source = trace.assessment_batches[0].source
    view = trace.assessment_batches[0].views[0]
    assert len(trace.assessment_batches) > 2
    assert data.count(source.source_json.encode()) == 1
    assert data.count(view.input_json.encode()) == 2  # Two typed views share it.
    restored = _decode_response(RunResponse, data).episode
    assert restored.to_record() == episode.to_record()
    assert restored.traces[0].assessment_batches == trace.assessment_batches


def test_proven_instances_are_exact_objects_bounded_by_retained_proofs(monkeypatch):
    from verifiers.v1 import _validation_scope as scopes

    source = _execution_source()
    with scopes.validation_scope():
        assert not scopes.proven_instance(source)
        source.verify()
        assert scopes.proven_instance(source)
        copied = source.model_copy()
        assert not scopes.proven_instance(copied)
        with pytest.raises(ValueError):
            source.model_copy(update={"source_digest": "tampered"}).verify()
        with pytest.raises(ValueError):
            source.model_copy(
                update={
                    "executions": (
                        source.executions[0].model_copy(update={"event_count": True}),
                    )
                }
            ).verify()
        scopes._owner.get().proofs.clear()
        assert not scopes.proven_instance(source)
    assert not scopes.proven_instance(source)
    assert not scopes._immutable((list, ((int, 1),)))
    assert not scopes._immutable((dict, ()))
    assert scopes._immutable(scopes._typed(source))


def test_batch_membership_index_matches_tuple_scan():
    from verifiers.v1.assessments import _membership

    source = _execution_source(second=True)
    contains = _membership(source.executions)
    assert all(contains(ref) for ref in source.executions)
    assert not contains(source.executions[0].model_copy(update={"phase": "before"}))
    unhashable = _membership(({"a": 1}, 2))
    assert unhashable({"a": 1}) and unhashable(2) and not unhashable(3)
    assert _membership((1, 2))([1]) is False


def test_occurrence_digests_keep_copied_coordinate_types_apart():
    ref = _execution_source().executions[0]
    a = vf.ExecutionRef.model_construct(**(ref.__dict__ | {"invocation_id": "1"}))
    b = vf.ExecutionRef.model_construct(**(ref.__dict__ | {"invocation_id": 1}))
    c = vf.ExecutionRef.model_construct(**(ref.__dict__ | {"invocation_id": True}))
    assert len({a.occurrence_id, b.occurrence_id, c.occurrence_id}) == 3
    assert ref.occurrence_id == vf.assessments.content_digest(
        {
            "episode_id": ref.episode_id,
            "trace_id": ref.trace_id,
            "origin": ref.origin,
            "invocation_id": ref.invocation_id,
        }
    )
