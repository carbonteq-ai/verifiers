"""Pluggable judges: plugin resolution, base-`TaskConfig.judges` narrowing, the built-in
`reference` / `rubric` judges, and `Task.score` running plugged judges after the decorated
rewards. Judge model calls are faked at `Judge.complete` — no network."""

import asyncio
import json
import re
import uuid
from types import SimpleNamespace

import pytest
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge import JudgeTaskConfig, ScoreConfig
from verifiers.v1.envs.agentic_judge.env import TRACE_FILE, JudgeTask
from verifiers.v1.graph import MessageNode
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.loaders import judge_class, judge_config_type, load_judge

RUBRIC_TOML = """
[[criteria]]
name = "mentions_paris"
text = "The response mentions Paris."
weight = 3.0

[[criteria]]
name = "is_polite"
text = "The response is polite."
"""


@pytest.mark.asyncio
async def test_custom_assessor_uses_raw_context_upstream_and_multiple_invocations():
    from verifiers.v1.assessment_runtime import execute_assessment

    source = vf.SourceSnapshot.capture(
        {"history": ["read", "send"], "state": {"excluded": True}},
        episode_id="episode",
        trace_ids=("trace",),
    )
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id="trace",
    )
    signal = vf.SignalDefinition(
        signal_id="harm",
        revision="1",
        semantics="cost",
        description="Verified harm",
        units="indicator",
    )
    views = tuple(
        vf.ObservationView.capture(
            value,
            snapshot_id=source.snapshot_id,
            builder_revision="raw@1",
            scope="retrospective",
            subjects=(subject,),
        )
        for value in ({"history": ["read", "send"]}, {"excluded": True})
    )
    parent = vf.Assessment(
        assessment_id="upstream",
        run_id="upstream-run",
        subject=subject,
        view_id=views[1].view_id,
        signal=signal,
        status="valid",
        value=1,
    )
    run = vf.AssessmentRun(
        run_id="custom",
        producer_id="custom",
        producer_revision="1",
        rubric_revision="1",
        snapshot_id=source.snapshot_id,
        invocation_id="outer",
        attempt_id="attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
    )
    request = vf.AssessmentRequest(source=source.identity, run=run, views=views)

    async def custom(request, ctx):
        assert ctx.dependencies == (parent,)
        history = ctx.input(views[0].view_id)
        history["history"].reverse()
        assert ctx.input(views[0].view_id)["history"] == ["read", "send"]
        assert ctx.input(views[1].view_id)["excluded"] is True
        for invocation in ("first", "second"):
            ctx.record_evidence(
                "custom_request", {"messages": history}, invocation_id=invocation
            )
            await asyncio.sleep(0)
            ctx.record_evidence(
                "custom_response",
                {"text": "harmful", "usage": 2},
                invocation_id=invocation,
            )
        yield vf.Assessment(
            assessment_id="finding",
            run_id=request.run.run_id,
            subject=subject,
            view_id=views[1].view_id,
            signal=signal,
            status="valid",
            value=1,
            invocation_ids=("first", "second"),
            derivation=vf.Derivation(
                rule_id="interpret",
                rule_revision="1",
                required_parent_ids=("upstream",),
            ),
        )

    retained = []
    result = await execute_assessment(
        custom, request, source, retained, dependencies=(parent,)
    )
    assert result.run.status == "complete"
    assert len(result.run.execution_evidence) == 4
    assert result.assessments[0].invocation_ids == ("first", "second")
    assert vf.AssessmentBatch.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("relation", ["preferred", "equivalent", "incomparable"])
def test_preference_publication_preserves_relation_without_numeric_conversion(relation):
    source = vf.SourceSnapshot.capture(
        {}, episode_id="episode", trace_ids=("left", "right")
    )
    alternatives = tuple(
        vf.SubjectRef(
            kind="trace",
            snapshot_id=source.snapshot_id,
            episode_id=source.episode_id,
            trace_id=trace_id,
        )
        for trace_id in source.trace_ids
    )
    subject = vf.SubjectRef(
        kind="group",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        members=alternatives,
    )
    signal = vf.SignalDefinition(
        signal_id="choice",
        revision="1",
        semantics="preference",
        description="Compare retained candidates",
        units="relation",
    )
    view = vf.ObservationView.capture(
        {"candidates": ["a", "b"]},
        snapshot_id=source.snapshot_id,
        builder_revision="1",
        scope="retrospective",
        subjects=(subject,),
    )
    run = vf.AssessmentRun(
        run_id="run",
        producer_id="custom",
        producer_revision="1",
        rubric_revision="1",
        snapshot_id=source.snapshot_id,
        invocation_id="outer",
        attempt_id="attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
    )
    finding = vf.Assessment(
        assessment_id="finding",
        run_id="run",
        subject=subject,
        view_id=view.view_id,
        signal=signal,
        status="valid",
        preference=vf.PreferenceResult(
            alternatives=alternatives,
            relation=relation,
            preferred_subject_id=alternatives[0].subject_id
            if relation == "preferred"
            else None,
        ),
    )
    batch = vf.AssessmentBatch(
        source=source, run=run, views=(view,), assessments=(finding,)
    )
    assert vf.AssessmentBatch.model_validate_json(batch.model_dump_json()) == batch
    assert finding.value is None
    with pytest.raises(ValueError):
        vf.Assessment.model_validate(finding.model_dump() | {"value": 0.5})
    with pytest.raises(ValueError):
        vf.Assessment.model_validate(finding.model_dump() | {"status": "failed"})
    with pytest.raises(ValueError):
        vf.PreferenceResult(
            alternatives=alternatives,
            relation="preferred",
            preferred_subject_id="absent",
        )
    with pytest.raises(ValueError, match="ordered group"):
        vf.Assessment.model_validate(
            finding.model_dump()
            | {
                "preference": vf.PreferenceResult(
                    alternatives=tuple(reversed(alternatives)), relation="equivalent"
                ).model_dump()
            }
        )


@pytest.mark.asyncio
async def test_prefix_context_rejects_later_views_and_unqualified_upstream():
    from verifiers.v1.assessment_runtime import (
        execute_assessment,
        execute_assessment_plan,
    )

    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id="trace",
    )
    signal = vf.SignalDefinition(
        signal_id="quality",
        revision="1",
        semantics="state_quality",
        description="Prefix quality",
        units="score",
    )
    prefix = vf.ObservationView.capture(
        {"history": ["read"]},
        snapshot_id=source.snapshot_id,
        builder_revision="1",
        scope="prefix",
        subjects=(subject,),
    )
    later = vf.ObservationView.capture(
        {"history": ["read", "send"]},
        snapshot_id=source.snapshot_id,
        builder_revision="1",
        scope="retrospective",
        subjects=(subject,),
    )
    run = vf.AssessmentRun(
        run_id="run",
        producer_id="custom",
        producer_revision="1",
        rubric_revision="1",
        snapshot_id=source.snapshot_id,
        invocation_id="outer",
        attempt_id="attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
    )
    with pytest.raises(ValueError, match="prefix context"):
        vf.AssessmentRequest(source=source.identity, run=run, views=(prefix, later))
    request = vf.AssessmentRequest(source=source.identity, run=run, views=(prefix,))
    parent = vf.Assessment(
        assessment_id="parent",
        run_id="parent-run",
        subject=subject,
        view_id=later.view_id,
        signal=signal,
        status="valid",
        value=1,
    )

    async def custom(request, ctx):
        assert ctx.input(prefix.view_id) == {"history": ["read"]}
        with pytest.raises(KeyError):
            ctx.input(later.view_id)
        yield vf.Assessment(
            assessment_id="finding",
            run_id="run",
            subject=subject,
            view_id=prefix.view_id,
            signal=signal,
            status="valid",
            value=0,
        )

    retained = []
    with pytest.raises(ValueError, match="upstream"):
        await execute_assessment(
            custom, request, source, retained, dependencies=(parent,)
        )
    assert retained == []
    dispatched = []

    async def external_call(request, ctx):
        dispatched.append(request.run.run_id)
        return ()

    full_run = run.model_copy(
        update={"run_id": "full", "attempt_id": "full", "invocation_id": "full"}
    )
    full_request = vf.AssessmentRequest(
        source=source.identity, run=full_run, views=(later,)
    )
    with pytest.raises(ValueError, match="upstream"):
        await execute_assessment_plan(
            {"full": external_call, "prefix": custom},
            [("full", full_request), ("prefix", request)],
            source,
            retained,
            max_concurrent=2,
            dependencies={request.run.attempt_id: (parent,)},
        )
    assert dispatched == []
    assert retained == []
    result = await execute_assessment(custom, request, source, retained)
    assert result.assessments[0].value == 0


class QAData(vf.TaskData):
    answer: str = ""


@pytest.mark.asyncio
async def test_credit_plan_rejects_changed_parents_before_any_dispatch():
    from verifiers.v1.credit import execute_credit_plan

    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id="trace",
    )
    signal = vf.SignalDefinition(
        signal_id="outcome",
        revision="1",
        semantics="outcome",
        description="Outcome",
        units="score",
    )
    parent = vf.Assessment(
        assessment_id="parent",
        run_id="run",
        subject=subject,
        view_id="view",
        signal=signal,
        status="valid",
        value=1,
    )

    def make_request(attempt, record):
        return vf.CreditRequest(
            source=source.identity,
            invocation_id=attempt,
            attempt_id=attempt,
            rule=vf.CreditRule(rule_id="identity", revision="1"),
            accepted=(record,),
            targets=(vf.CreditTarget(recipient=subject),),
            allocation="turn_boundary",
            overlap_policy="reject",
        )

    dispatched = []

    async def rule(request):
        dispatched.append(request.attempt_id)
        yield vf.CreditContribution(
            contribution_id="credit",
            parent_assessment_ids=("parent",),
            recipient=subject,
            signal=signal,
            transformation="identity",
            status="valid",
            value=1,
            allocation="turn_boundary",
            attribution="coarse",
        )

    good = make_request("good", parent)
    for changed in (
        parent.model_copy(update={"value": 0}),
        parent.model_copy(update={"assessment_id": "invented"}),
    ):
        retained = []
        with pytest.raises(ValueError, match="unretained or changed"):
            await execute_credit_plan(
                {"first": rule, "second": rule},
                [("first", good), ("second", make_request("bad", changed))],
                source,
                retained,
                accepted=(parent,),
            )
        assert not dispatched and not retained
    retained = []
    await execute_credit_plan(
        {"first": rule, "unused": rule},
        [("first", good)],
        source,
        retained,
        accepted=(parent,),
    )
    assert isinstance(retained[-1].request.source, vf.SourceSnapshot)
    with pytest.raises(ValueError, match="unregistered hooks"):
        await execute_credit_plan(
            {"first": rule},
            [("unknown", make_request("unknown", parent))],
            source,
            [],
            accepted=(parent,),
        )


@pytest.mark.asyncio
async def test_registered_credit_rule_can_have_no_applicable_assignments():
    from verifiers.v1.credit import execute_credit_plan

    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    retained = []

    async def rule(request):
        raise AssertionError("empty credit plan must not dispatch")

    await execute_credit_plan({"rule": rule}, [], source, retained, accepted=())
    assert retained == []


@pytest.mark.asyncio
async def test_credit_executor_seals_request_against_plugin_mutation():
    source = vf.SourceSnapshot.capture({}, episode_id="episode", trace_ids=("trace",))
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id="trace",
    )
    signal = vf.SignalDefinition(
        signal_id="outcome",
        revision="1",
        semantics="outcome",
        description="Outcome",
        units="score",
    )
    parent = vf.Assessment(
        assessment_id="parent",
        run_id="run",
        subject=subject,
        view_id="view",
        signal=signal,
        status="valid",
        value=1,
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="invocation",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="identity", revision="1"),
        accepted=(parent,),
        targets=(vf.CreditTarget(recipient=subject),),
        allocation="turn_boundary",
        overlap_policy="reject",
    )

    async def rule(request):
        request.accepted[0].__dict__["value"] = 0
        yield vf.CreditContribution(
            contribution_id="credit",
            parent_assessment_ids=("parent",),
            recipient=subject,
            signal=signal,
            transformation="identity",
            status="valid",
            value=0,
            allocation="turn_boundary",
            attribution="coarse",
        )

    retained = []
    result = await vf.execute_credit_assignment(rule, request, retained)
    assert result.status == "failed"
    assert all(item.request.accepted[0].value == 1 for item in retained)
    assert request.accepted[0].value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["valid", "failure", "cancel", "wrong_parent", "missing"]
)
async def test_native_assignment_executor_retains_channels_and_partial_failures(mode):
    source = vf.SourceSnapshot.capture(
        {"raw": "tokenless"}, episode_id="episode", trace_ids=("trace",)
    )
    recipient = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id=source.episode_id,
        trace_id="trace",
    )
    raw_signal = vf.SignalDefinition(
        signal_id="violation",
        revision="1",
        semantics="outcome",
        description="Verified violation",
        units="indicator",
    )
    cost_signal = vf.SignalDefinition(
        signal_id="guard-cost",
        revision="1",
        semantics="cost",
        description="Signed guard cost",
        units="cost",
    )
    parent = vf.Assessment(
        assessment_id="raw",
        run_id="run",
        subject=recipient,
        view_id="view",
        signal=raw_signal,
        status="valid",
        value=1,
    )
    request = vf.CreditRequest(
        source=source,
        invocation_id="invocation",
        attempt_id="attempt",
        rule=vf.CreditRule(rule_id="assign-violation", revision="1"),
        accepted=(parent,),
        targets=(
            vf.CreditTarget(recipient=recipient, channel="guard"),
            vf.CreditTarget(recipient=recipient, channel="outcome"),
        ),
        allocation="turn_boundary",
        overlap_policy="reject",
    )

    async def rule(request):
        if mode == "missing":
            return
        yield vf.CreditContribution(
            contribution_id="guard",
            parent_assessment_ids=("wrong" if mode == "wrong_parent" else "raw",),
            recipient=recipient,
            channel="guard",
            signal=cost_signal,
            transformation="violation-to-cost@1",
            status="valid",
            value=-1,
            allocation="turn_boundary",
            attribution="coarse",
        )
        if mode == "failure":
            raise RuntimeError("PRIVATE-FAILURE")
        if mode == "cancel":
            raise asyncio.CancelledError
        yield vf.CreditContribution(
            contribution_id="outcome",
            parent_assessment_ids=("raw",),
            recipient=recipient,
            channel="outcome",
            signal=raw_signal,
            transformation="identity",
            status="valid",
            value=1,
            allocation="turn_boundary",
            attribution="coarse",
        )

    retained = []
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await vf.execute_credit_assignment(rule, request, retained)
    else:
        await vf.execute_credit_assignment(rule, request, retained)
    result = retained[-1]
    assert result.status == (
        "complete"
        if mode == "valid"
        else "interrupted"
        if mode == "cancel"
        else "failed"
    )
    assert len(result.contributions) == (
        2 if mode == "valid" else 1 if mode in {"failure", "cancel"} else 0
    )
    assert result.request.accepted[0].value == 1
    assert "PRIVATE-FAILURE" not in result.model_dump_json()
    assert vf.CreditAssignment.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValueError, match="already"):
        await vf.execute_credit_assignment(rule, request, retained)


@pytest.mark.asyncio
async def test_native_assessment_lifecycle_keeps_scalar_and_independent_failures():
    started = set()
    both_started = asyncio.Event()

    class AssessedTask(vf.Task):
        fail_scalar = False
        cancel_scalar = False
        concurrent_probe = False
        use_planning_context = False
        fail_deterministic = False

        def plan_credit(self, source, assessments, context):
            assert context.source == source.identity
            if not self.use_planning_context:
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
                and batch.run.producer_id == "deterministic"
                and batch.run.status == "complete"
            )
            if not selected:
                return []
            if any(
                item.status == "complete"
                and item.request.rule.rule_id == "guard-cost"
                and item.request.source.snapshot_id == source.snapshot_id
                for item in context.prior_assignments
            ):
                # This fixture explicitly owns a once-per-source policy. The
                # runtime has neither dropped history nor deduplicated findings.
                return []
            return self.credit_requests(source, selected)

        async def rendezvous(self, name):
            if self.concurrent_probe:
                started.add(name)
                if len(started) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=2)

        @vf.reward
        async def outcome(self, trace):
            if self.cancel_scalar:
                raise asyncio.CancelledError
            if self.fail_scalar:
                raise ValueError("legacy scorer failed")
            return 0.75

        def assessment_requests(self, source):
            subject = vf.SubjectRef(
                kind="trace",
                snapshot_id=source.snapshot_id,
                episode_id=source.episode_id,
                trace_id=source.trace_ids[0],
            )
            signal = vf.SignalDefinition(
                signal_id="guard",
                revision="1",
                semantics="outcome",
                description="Verified guard violation",
                units="indicator",
            )
            result = []
            for hook_name in ("deterministic", "optional_model"):
                attempt = uuid.uuid4().hex
                view = vf.ObservationView.capture(
                    {"guard_violated": True},
                    snapshot_id=source.snapshot_id,
                    builder_revision="guard@1",
                    scope="retrospective",
                    subjects=(subject,),
                )
                run = vf.AssessmentRun(
                    run_id=attempt,
                    producer_id=hook_name,
                    producer_revision="1",
                    rubric_revision="1",
                    snapshot_id=source.snapshot_id,
                    invocation_id=attempt,
                    attempt_id=attempt,
                    expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
                )
                result.append(
                    (
                        hook_name,
                        vf.AssessmentRequest(
                            source=source.identity, run=run, views=(view,)
                        ),
                    )
                )
            return result

        @vf.assessment
        async def deterministic(self, request, ctx):
            await self.rendezvous("deterministic")
            if self.fail_deterministic:
                raise RuntimeError("current deterministic assessment failed")
            assert ctx.input(request.views[0].view_id)["guard_violated"] is True
            ctx.record_evidence("deterministic_input", {"guard_violated": True})
            target = request.run.expected[0]
            yield vf.Assessment(
                assessment_id=uuid.uuid4().hex,
                run_id=request.run.run_id,
                subject=target.subject,
                signal=target.signal,
                view_id=request.views[0].view_id,
                status="valid",
                value=1,
            )

        @vf.assessment
        async def optional_model(self, ctx):
            await self.rendezvous("optional_model")
            ctx.record_evidence(
                "received_response", {"text": "unusable provider response"}
            )
            raise RuntimeError("provider unavailable")

        def credit_requests(self, source, assessments):
            accepted = next(
                batch.assessments
                for batch in reversed(assessments)
                if batch.run.producer_id == "deterministic"
                and batch.run.status == "complete"
                and batch.source.snapshot_id == source.snapshot_id
            )
            attempt = uuid.uuid4().hex
            return [
                (
                    "guard_credit",
                    vf.CreditRequest(
                        source=source.identity,
                        invocation_id=attempt,
                        attempt_id=attempt,
                        rule=vf.CreditRule(rule_id="guard-cost", revision="1"),
                        accepted=accepted,
                        targets=(
                            vf.CreditTarget(
                                recipient=accepted[0].subject, channel="guard"
                            ),
                        ),
                        allocation="turn_boundary",
                        overlap_policy="reject",
                    ),
                )
            ]

        @vf.credit
        async def guard_credit(self, request):
            parent = request.accepted[0]
            yield vf.CreditContribution(
                contribution_id=uuid.uuid4().hex,
                parent_assessment_ids=(parent.assessment_id,),
                recipient=parent.subject,
                channel="guard",
                signal=vf.SignalDefinition(
                    signal_id="guard-cost",
                    revision="1",
                    semantics="cost",
                    description="Signed guard cost",
                    units="cost",
                ),
                transformation="violation-to-cost@1",
                status="valid",
                value=-parent.value,
                allocation="turn_boundary",
                attribution="coarse",
            )

    task = AssessedTask(vf.TaskData(prompt="question"))
    trace = make_trace()
    trace.episode_id = "native-episode"
    await task.score(trace)
    assert trace.reward == 0.75
    complete = [b for b in trace.assessment_batches if b.run.status == "complete"]
    failed = [b for b in trace.assessment_batches if b.run.status == "failed"]
    assert len(complete) == len(failed) == 1
    assert complete[0].assessments[0].value == 1
    assert len(failed[0].missing) == 1 and not failed[0].assessments
    assert complete[0].run.execution_evidence[0].kind == "deterministic_input"
    assert failed[0].run.execution_evidence[0].kind == "received_response"
    assert trace.credit_assignments[-1].status == "complete"
    assert trace.credit_assignments[-1].contributions[0].value == -1
    assert not trace.credit_errors
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert restored.assessment_batches == trace.assessment_batches
    assert restored.credit_assignments == trace.credit_assignments
    source_before = complete[0].source.snapshot_id
    await task.score(trace)
    assert sum(b.run.status == "complete" for b in trace.assessment_batches) == 2
    assert trace.assessment_batches[-1].source.snapshot_id == source_before
    deferred = task.defer_scoring()
    count = len(trace.assessment_batches)
    await deferred.score(trace)
    assert len(trace.assessment_batches) == count

    contextual = AssessedTask(vf.TaskData(prompt="question"))
    contextual.use_planning_context = True
    contextual.contexts = []
    current_trace = make_trace()
    current_trace.episode_id = "contextual-episode"
    await contextual.score(current_trace)
    first_assignments = list(current_trace.credit_assignments)
    first_context = contextual.contexts[0]
    assert not first_context.prior_assignments
    assert {run.producer_id for run in first_context.current_assessment_runs} == {
        "deterministic",
        "optional_model",
    }
    await contextual.score(current_trace)
    assert current_trace.credit_assignments == first_assignments
    assert contextual.contexts[-1].prior_assignments == tuple(first_assignments)
    assert {run.run_id for run in first_context.current_assessment_runs}.isdisjoint(
        run.run_id for run in contextual.contexts[-1].current_assessment_runs
    )
    contextual.fail_deterministic = True
    await contextual.score(current_trace)
    assert current_trace.credit_assignments == first_assignments
    assert not current_trace.credit_errors
    last_ids = {run.run_id for run in contextual.contexts[-1].current_assessment_runs}
    assert all(
        batch.run.status != "complete"
        for batch in current_trace.assessment_batches
        if batch.run.run_id in last_ids
    )
    # An older successful finding remains retained, but is not selected as a
    # substitute for this scoring call's failed producer.
    assert any(
        batch.run.status == "complete" for batch in current_trace.assessment_batches
    )
    assert (
        vf.CreditPlanningContext.model_validate_json(first_context.model_dump_json())
        == first_context
    )
    changed = first_context.model_dump(mode="json")
    changed["source"]["snapshot_id"] = "foreign-snapshot"
    with pytest.raises(ValueError, match="planning source mismatch"):
        vf.CreditPlanningContext.model_validate(changed)
    changed = first_context.model_dump(mode="json")
    changed["current_assessment_runs"].append(changed["current_assessment_runs"][0])
    with pytest.raises(ValueError, match="duplicate current assessment"):
        vf.CreditPlanningContext.model_validate(changed)
    foreign = vf.CreditPlanningContext(
        source=first_context.source.model_copy(
            update={"episode_id": "foreign-episode"}
        ),
        current_assessment_runs=(),
    )
    with pytest.raises(ValueError, match="context/source mismatch"):
        vf.Task.plan_credit(
            contextual, current_trace.assessment_batches[0].source, (), foreign
        )
    assert sum(item.status == "complete" for item in trace.credit_assignments) == 2
    task.fail_scalar = task.cancel_scalar = False
    task.concurrent_probe = True
    await task.score(trace)
    assert started == {"deterministic", "optional_model"}
    assert sum(b.run.status == "complete" for b in trace.assessment_batches) == 3
    task.concurrent_probe = False
    task.fail_scalar = True
    with pytest.raises(vf.TaskError):
        await task.score(trace)
    assert sum(b.run.status == "complete" for b in trace.assessment_batches) == 4
    assert sum(item.status == "complete" for item in trace.credit_assignments) == 4
    task.cancel_scalar = True
    count = len(trace.assessment_batches)
    with pytest.raises(asyncio.CancelledError):
        await task.score(trace)
    assert len(trace.assessment_batches) == count


def make_trace(
    reply: str = "It is Paris.",
    answer: str = "Paris",
    task_cls: type[QAData] = QAData,
) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=task_cls(idx=0, prompt="Capital of France?", answer=answer),
        ),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content="Capital of France?"),
                sampled=False,
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content=reply), sampled=True
            ),
        ],
    )


@pytest.fixture
def fake_judge_model(monkeypatch):
    """Fake the judge's model call, recording each prompt for assertions. Rubric calls (a JSON
    `verdicts` instruction or a `schema`) reply one reasoned verdict per `- name: text` criterion,
    "yes" iff it mentions Paris; other judges reply plain "yes"/"no" by the response block."""
    prompts: list[str] = []

    async def fake_complete(
        self, messages, *, trace=None, schema=None, parse=None, **sampling
    ):
        prompts.append(messages)
        # Rubric calls carry criteria lines + a JSON `verdicts` instruction; reply one verdict per
        # criterion (yes iff its text mentions Paris) with a reason. Other judges get plain yes/no.
        if schema is not None or '"verdicts"' in messages:
            verdicts = [
                {
                    "name": name,
                    "reason": "cites Paris" if "Paris" in text else "no Paris",
                    "verdict": "yes" if "Paris" in text else "no",
                }
                for name, text in re.findall(
                    r"^- ([^:]+): (.+)$", messages, re.MULTILINE
                )
            ]
            response = JudgeResponse(
                text=json.dumps({"verdicts": verdicts}),
                parsed=schema.model_validate({"verdicts": verdicts})
                if schema
                else None,
            )
        else:
            response = JudgeResponse(
                text="yes" if "Paris" in messages.split("Response:")[-1] else "no"
            )
        if parse is not None:
            response.parsed = parse(response)
        if trace is not None:
            from verifiers.v1.dialects.chat import message_to_wire

            wire = (
                [{"role": "user", "content": messages}]
                if isinstance(messages, str)
                else [message_to_wire(message) for message in messages]
            )
            kwargs = {"model": self.config.model, "messages": wire}
            kwargs.update(sampling)
            trace.record_judge_call(
                name=self.reward_name,
                request={"model": kwargs["model"], "messages": kwargs["messages"]},
                response=response,
            )
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)
    return prompts


# --- plugin resolution + config narrowing --------------------------------------------------


def test_judge_plugin_resolution():
    assert judge_class("reference") is vf.ReferenceJudge
    assert judge_class("rubric") is vf.RubricJudge
    assert judge_config_type("reference") is vf.ReferenceJudgeConfig
    assert judge_config_type("rubric") is vf.RubricJudgeConfig
    assert isinstance(load_judge(vf.ReferenceJudgeConfig()), vf.ReferenceJudge)
    # built-ins pin their id, so a code-level default entry needs no explicit id
    assert vf.ReferenceJudgeConfig().id == "reference"
    assert vf.RubricJudgeConfig(path="x.toml").id == "rubric"

    # a judge without the config generic falls back to the base JudgeConfig
    class PlainJudge(vf.Judge[bool]):
        prompt = "{x}"

    assert type(PlainJudge().config) is vf.JudgeConfig
    assert type(vf.ReferenceJudge().config) is vf.ReferenceJudgeConfig


def test_taskset_config_narrows_judges(tmp_path):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = vf.TaskConfig.model_validate(
        {
            "judges": [
                {"id": "reference", "answer_field": "gold", "weight": 0.5},
                {"id": "rubric", "path": str(rubric), "name": "quality"},
            ]
        }
    )
    reference, rubric_cfg = cfg.judges
    assert (
        isinstance(reference, vf.ReferenceJudgeConfig)
        and reference.answer_field == "gold"
    )
    assert isinstance(rubric_cfg, vf.RubricJudgeConfig) and rubric_cfg.name == "quality"
    assert cfg.model_dump()["judges"][0]["answer_field"] == "gold"
    again = vf.TaskConfig.model_validate(cfg.model_dump())
    assert isinstance(again.judges[0], vf.ReferenceJudgeConfig)


def test_judges_entry_requires_id():
    with pytest.raises(ValueError, match="needs an `id`"):
        vf.TaskConfig.model_validate({"judges": [{"weight": 1.0}]})


def test_rubric_config_requires_path():
    # `path` is a required Path field: a plugged rubric judge without one fails at config time.
    with pytest.raises(ValueError, match="path"):
        vf.TaskConfig.model_validate({"judges": [{"id": "rubric"}]})


def test_judges_reject_shared_reward_keys():
    # Ids may repeat (same plugin, two configs) — what must be unique is the derived reward
    # key (`name`, else the id's package name), checked at config time.
    with pytest.raises(ValueError, match="share a reward key"):
        vf.TaskConfig.model_validate(
            {"judges": [{"id": "reference"}, {"id": "reference"}]}
        )
    cfg = vf.TaskConfig.model_validate(
        {
            "judges": [
                {"id": "reference", "name": "strict"},
                {"id": "reference", "name": "lenient"},
            ]
        }
    )
    assert [judge.name for judge in cfg.judges] == ["strict", "lenient"]

    # class-level DEFAULTS are held to the same rule (they bypass the before-hook)
    class TwoDefaults(vf.TaskConfig):
        judges: vf.Judges = Field(
            default_factory=lambda: [
                vf.ReferenceJudgeConfig(),
                vf.ReferenceJudgeConfig(),
            ]
        )

    with pytest.raises(ValueError, match="share a reward key"):
        TwoDefaults()


def test_reward_name_fallback():
    # name > id > snake-cased class name (code-level judge with neither).
    class MyQualityJudge(vf.Judge[float]):
        prompt = "{x}"

    assert (
        vf.ReferenceJudge().reward_name == "reference"
    )  # class-name fallback (no id set)
    assert (
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="my-judge")).reward_name
        == "my-judge"
    )
    assert vf.ReferenceJudge(vf.ReferenceJudgeConfig(name="gold")).reward_name == "gold"
    assert MyQualityJudge().reward_name == "my_quality"


async def test_base_judge_score_raises():
    with pytest.raises(NotImplementedError, match="implements no `score`"):
        await vf.Judge().score(task=QAData(idx=0, prompt="q"), trace=make_trace())


# --- reference --------------------------------------------------------------------------------


def test_reference_parse():
    judge = vf.ReferenceJudge()
    assert judge.parse(JudgeResponse(text="yes")) == 1.0
    assert judge.parse(JudgeResponse(text="Final answer: NO")) == 0.0
    # an unparseable verdict is a judge failure: raise (-> rollout error), don't score 0
    with pytest.raises(ValueError, match="no yes/no verdict"):
        judge.parse(JudgeResponse(text="gibberish"))


async def test_reference_score(fake_judge_model):
    trace = make_trace()
    verdict = await vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="reference")).score(
        trace.task.data, trace
    )
    assert verdict == 1.0
    assert (
        "Capital of France?" in fake_judge_model[0]
    )  # the task prompt is in the judge prompt
    assert len(trace.info["judge_calls"]) == 1  # the call is recorded onto the trace
    judge_record = trace.info["judge_calls"][0]
    assert judge_record["name"] == "reference"
    assert judge_record["request"]["model"] == "openai/gpt-5.4-nano"
    assert "Capital of France?" in judge_record["request"]["messages"][0]["content"]
    assert judge_record["response"]["message"] == {
        "role": "assistant",
        "content": "yes",
        "reasoning_content": None,
        "tool_calls": None,
        "provider_state": None,
    }
    assert judge_record["response"]["parsed"] == 1.0

    request_messages = [{"role": "user", "content": "original"}]
    request = {
        "model": "openai/gpt-5.4-nano",
        "messages": request_messages,
    }
    trace.record_judge_call(
        name="snapshot",
        request=request,
        response=JudgeResponse(text="yes"),
    )
    request_messages[0]["content"] = "mutated"
    assert (
        trace.info["judge_calls"][-1]["request"]["messages"][0]["content"] == "original"
    )

    override_trace = make_trace()
    await vf.ReferenceJudge().complete(
        "Judge this response.", trace=override_trace, model="openai/gpt-5.4-mini"
    )
    assert (
        override_trace.info["judge_calls"][0]["request"]["model"]
        == "openai/gpt-5.4-mini"
    )

    trace = make_trace(reply="It is Rome.")
    assert await vf.ReferenceJudge().score(trace.task.data, trace) == 0.0

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="gold"))
    with pytest.raises(ValueError, match="no 'gold' field"):  # misconfig raises, not 0
        await judge.score(trace.task.data, trace)


async def test_reference_score_messages_prompt(fake_judge_model):
    # A Messages-form prompt still reaches the judge as text (via TaskData.prompt_text).
    from verifiers.v1.types import TextContentPart
    from verifiers.v1.types import UserMessage as UM

    task = QAData(
        idx=0,
        prompt=[UM(content=[TextContentPart(text="Capital of France?")])],
        answer="Paris",
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    assert await vf.ReferenceJudge().score(task, trace) == 1.0
    assert "Capital of France?" in fake_judge_model[0]


async def test_reference_question_field(fake_judge_model):
    # question_field points {question} at a dedicated task field instead of the full prompt.
    class FieldTask(vf.TaskData):
        question: str = ""
        answer: str = ""

    task = FieldTask(
        idx=0,
        prompt="SYSTEM INSTRUCTIONS\n\nQuestion: Capital of France?",
        question="Capital of France?",
        answer="Paris",
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="question")).score(
        task, trace
    )
    assert "Capital of France?" in fake_judge_model[0]
    assert "SYSTEM INSTRUCTIONS" not in fake_judge_model[0]

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="typo"))
    with pytest.raises(ValueError, match="no 'typo' field"):
        await judge.score(task, trace)


def full_trace_fixture() -> vf.Trace:
    """A multi-turn trace: user -> assistant (reasoning + tool call) -> tool -> final reply."""
    from verifiers.v1.types import ToolCall, ToolMessage

    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task", data=QAData(idx=0, prompt="Capital of France?", answer="Paris")
        ),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content="Capital of France?"),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="Let me look it up.",
                    reasoning_content="SECRET REASONING",
                    provider_state=[{"type": "reasoning", "data": "SECRET STATE"}],
                    tool_calls=[
                        ToolCall(id="1", name="search", arguments='{"q": "france"}')
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=1,
                message=ToolMessage(
                    tool_call_id="1",
                    name="search",
                    content="TOOL RESULT: Paris is the capital.",
                ),
                sampled=False,
            ),
            MessageNode(
                parent=2, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )


def test_transcript():
    transcript = full_trace_fixture().transcript
    assert "[user]\nCapital of France?" in transcript
    assert '[tool_call search({"q": "france"})]' in transcript
    assert "[tool search]\nTOOL RESULT: Paris is the capital." in transcript
    assert "It is Paris." in transcript
    assert "SECRET REASONING" not in transcript  # reasoning is excluded


def test_agentic_judge_trace_hidden_reasoning_toggle():
    task = JudgeTask.from_trace(full_trace_fixture(), JudgeTaskConfig())
    record = json.loads(task.files[TRACE_FILE])
    assistant = record["nodes"][1]["message"]
    assert assistant["content"] == "Let me look it up."
    assert assistant["tool_calls"][0]["name"] == "search"
    assert "reasoning_content" not in assistant
    assert "provider_state" not in assistant

    task = JudgeTask.from_trace(
        full_trace_fixture(), JudgeTaskConfig(include_hidden_reasoning=True)
    )
    assistant = json.loads(task.files[TRACE_FILE])["nodes"][1]["message"]
    assert assistant["reasoning_content"] == "SECRET REASONING"
    assert assistant["provider_state"] == [
        {"type": "reasoning", "data": "SECRET STATE"}
    ]


async def test_view_modes(fake_judge_model):
    # last_reply (default): the judge sees only the final reply.
    trace = full_trace_fixture()
    await vf.ReferenceJudge().score(trace.task.data, trace)
    assert "TOOL RESULT" not in fake_judge_model[0]
    # full_trace: the whole transcript (minus reasoning) fills {response}.
    trace = full_trace_fixture()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(view="full_trace")).score(
        trace.task.data, trace
    )
    assert "TOOL RESULT: Paris is the capital." in fake_judge_model[1]
    assert "SECRET REASONING" not in fake_judge_model[1]


async def test_rubric_view_full_trace(tmp_path, fake_judge_model):
    # The rubric judge's default view: criteria are judged against the whole transcript.
    judge = rubric_judge(tmp_path)
    trace = full_trace_fixture()
    await judge.score(trace.task.data, trace)
    assert all("TOOL RESULT" in prompt for prompt in fake_judge_model)
    assert all("SECRET REASONING" not in prompt for prompt in fake_judge_model)


async def test_config_prompt_overrides_class_template(tmp_path, fake_judge_model):
    # Config `prompt` is a file path; the same {field} placeholders work.
    file = tmp_path / "judge.txt"
    file.write_text("Q:{question} A:{answer} R:{response}")
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt=file))
    assert judge.build_messages(question="q", answer="a", response="r") == "Q:q A:a R:r"
    # A template needn't use every evaluate field: score also passes {positive}/{negative},
    # which str.format ignores when the (custom) prompt doesn't reference them.
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 1.0
    assert fake_judge_model[0] == "Q:Capital of France? A:Paris R:It is Paris."
    # a bad path fails at judge construction, not mid-eval at score time
    with pytest.raises(FileNotFoundError):
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt=tmp_path / "missing.txt"))


# --- reference input/verdict knobs ------------------------------------------------------------


async def test_reference_list_answer(fake_judge_model):
    # A list-valued answer field is judged as multiple acceptable answers, one per line.
    class MultiTask(vf.TaskData):
        aliases: list[str] = Field(default_factory=list)

    task = MultiTask(idx=0, prompt="q?", aliases=["Paris", "Lutetia"])
    trace = make_trace()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="aliases")).score(
        task, trace
    )
    assert "Paris\nLutetia" in fake_judge_model[0]


async def test_reference_choices(fake_judge_model):
    # Verdict labels are configurable; the positive (first) label scores 1.0 and the
    # default prompt asks for the configured labels (not a hardcoded yes/no).
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(choices=("A", "B")))
    assert judge.parse(JudgeResponse(text="A")) == 1.0
    assert judge.parse(JudgeResponse(text="Final verdict: B")) == 0.0
    with pytest.raises(ValueError, match="no A/B verdict"):
        judge.parse(JudgeResponse(text="gibberish"))
    trace = make_trace()
    with pytest.raises(
        ValueError
    ):  # the yes-replying fake is now an unparseable verdict
        await judge.score(trace.task.data, trace)
    assert 'Respond either "A" or "B"' in fake_judge_model[0]
    # degenerate labels are a config error (duplicates would score every verdict 1.0)
    for choices in (("yes", "yes"), ("A", "a"), ("", "no")):
        with pytest.raises(ValueError, match="two distinct, non-empty"):
            vf.ReferenceJudgeConfig(choices=choices)


async def test_error_attribution(monkeypatch, tmp_path):
    # The policy: a MODEL failure scores 0.0; a JUDGE failure errors the rollout (raises
    # out of Task.score as a TaskError) so training skips the sample instead of
    # punishing the model for a broken judge.
    async def gibberish_judge(
        self, messages, *, trace=None, schema=None, parse=None, **s
    ):
        response = JudgeResponse(text="as an AI language model I cannot grade this")
        try:
            if parse is not None:
                response.parsed = parse(response)
            return response
        finally:
            if trace is not None:
                trace.record_judge_call(
                    name=self.reward_name,
                    request={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": messages}],
                    },
                    response=response,
                )

    monkeypatch.setattr(Judge, "complete", gibberish_judge)
    taskset = JudgedTaskset(
        JudgedConfig.model_validate({"task": {"judges": [{"id": "reference"}]}})
    )
    # model failure: empty reply -> judge skipped, reward 0.0, NO error
    trace = make_trace(reply="")
    await JudgedTask(trace.task.data, taskset.config.task).score(trace, runtime=None)
    assert trace.rewards["reference"].score == 0.0
    assert "judge_calls" not in trace.info  # the (foregone) judge call was never made
    # judge failure: unparseable verdict -> the rollout errors, no reward recorded
    trace = make_trace()
    with pytest.raises(vf.TaskError, match="no yes/no verdict"):
        await JudgedTask(trace.task.data, taskset.config.task).score(
            trace, runtime=None
        )
    assert trace.rewards["reference"] is None  # seeded: expected but never scored
    assert len(trace.info["judge_calls"]) == 1  # the billed call is still recorded


# --- rubric --------------------------------------------------------------------------------


def rubric_judge(
    tmp_path, body: str = RUBRIC_TOML, suffix: str = ".toml", **kwargs
) -> vf.RubricJudge:
    path = tmp_path / f"rubric{suffix}"
    path.write_text(body)
    return vf.RubricJudge(vf.RubricJudgeConfig(path=str(path), **kwargs))


def test_rubric_criteria_toml_and_json(tmp_path):
    toml = rubric_judge(tmp_path).criteria
    assert [c.name for c in toml] == ["mentions_paris", "is_polite"]
    assert [c.weight for c in toml] == [3.0, 1.0]
    # JSON accepts a metadata-bearing object or a bare criteria list.
    items = [c.model_dump() for c in toml]
    assert (
        rubric_judge(
            tmp_path,
            json.dumps({"title": "Safety", "criteria": items}),
            ".json",
        ).criteria
        == toml
    )
    assert rubric_judge(tmp_path, json.dumps(items), ".json").criteria == toml
    # the suffix check is case-insensitive: QUALITY.TOML is TOML, not JSON
    assert rubric_judge(tmp_path, suffix=".TOML").criteria == toml


def test_rubric_config_weight_overrides_file(tmp_path):
    judge = rubric_judge(tmp_path, weights={"mentions_paris": 1.0})
    assert [c.weight for c in judge.criteria] == [1.0, 1.0]


def test_rubric_rejects_bad_files(tmp_path):
    with pytest.raises(ValueError, match="no criteria"):
        _ = rubric_judge(tmp_path, "").criteria
    duplicate = RUBRIC_TOML.replace("is_polite", "mentions_paris")
    with pytest.raises(ValueError, match="duplicate"):
        _ = rubric_judge(tmp_path, duplicate).criteria
    with pytest.raises(ValueError, match="name no criterion"):
        _ = rubric_judge(tmp_path, weights={"typo": 2.0}).criteria
    # all-zero weights fail while loading the rubric — before any judge call is paid for
    with pytest.raises(ValueError, match="no positive criterion weight"):
        _ = rubric_judge(
            tmp_path, weights={"mentions_paris": 0.0, "is_polite": 0.0}
        ).criteria
    # negative/NaN/inf weights would invert a criterion or corrupt the weighted mean
    for weight in (-1.0, float("nan"), float("inf")):
        with pytest.raises(
            ValueError, match="greater than or equal to 0|finite number"
        ):
            _ = rubric_judge(tmp_path, weights={"mentions_paris": weight}).criteria
    with pytest.raises(ValueError, match="non-finite total"):
        _ = rubric_judge(
            tmp_path, weights={"mentions_paris": 1e308, "is_polite": 1e308}
        ).criteria


def test_judge_composition_weights_are_finite():
    for weight in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            vf.JudgeConfig(weight=weight)
        for field in ("task_weight", "judge_weight"):
            with pytest.raises(ValueError, match="finite number"):
                ScoreConfig.model_validate({field: weight})


async def test_rubric_score(tmp_path, fake_judge_model):
    # verdicts: mentions_paris=1 (w=3), is_polite=0 (w=1) -> weighted mean 0.75, from ONE
    # judge call; each verdict lands as a `<name>/<criterion>` metric.
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.75
    assert trace.metrics == {"rubric/mentions_paris": 1.0, "rubric/is_polite": 0.0}
    assert len(trace.info["judge_calls"]) == 1  # one call for the whole rubric


async def test_rubric_verdict_mismatch_raises(tmp_path, monkeypatch):
    # A reply that doesn't verdict exactly the rubric's criteria is a judge failure: raise
    # (-> rollout error), don't guess or silently score 0.
    async def wrong_names(self, messages, *, trace=None, schema=None, parse=None, **s):
        verdicts = {"verdicts": [{"name": "typo", "reason": "x", "verdict": "yes"}]}
        return JudgeResponse(text=json.dumps(verdicts), parsed=None)

    monkeypatch.setattr(Judge, "complete", wrong_names)
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected the batch"):
        await judge.score(trace.task.data, trace)


CHOICES_TOML = '[[criteria]]\nname = "depth"\ntext = "How thorough?"\nchoices = ["none", "partial", "good"]\n'


async def test_rubric_choices_normalize(tmp_path, monkeypatch):
    # Ordered choices (worst→best) score by rank: "partial" of ["none","partial","good"] -> 0.5.
    async def graded(self, messages, *, trace=None, schema=None, parse=None, **s):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "partial"}]}
        return JudgeResponse(text=json.dumps(v), parsed=None)

    monkeypatch.setattr(Judge, "complete", graded)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML, name="q")
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.5
    assert trace.metrics == {"q/depth": 0.5}


async def test_rubric_off_menu_answer_raises(tmp_path, monkeypatch):
    # A verdict that isn't one of the criterion's choices is a judge failure, not a 0.
    async def off_menu(self, messages, *, trace=None, schema=None, parse=None, **s):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "maybe"}]}
        return JudgeResponse(text=json.dumps(v), parsed=None)

    monkeypatch.setattr(Judge, "complete", off_menu)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected one of"):
        await judge.score(trace.task.data, trace)


def test_rubric_choices_validation(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        _ = rubric_judge(
            tmp_path, body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["only"]\n'
        ).criteria
    with pytest.raises(ValueError, match="duplicate options"):
        _ = rubric_judge(
            tmp_path,
            body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["a", "a"]\n',
        ).criteria


async def test_rubric_reference_answer_optional(tmp_path, fake_judge_model):
    # off by default: no reference block in the prompt.
    t = make_trace()
    await rubric_judge(tmp_path).score(t.task.data, t)
    assert "Reference solution" not in fake_judge_model[-1]

    # answer_field set: the task's gold answer is shown to the judge.
    t = make_trace(answer="ZEBRA-GOLD")
    await rubric_judge(tmp_path, answer_field="answer").score(t.task.data, t)
    assert "Reference solution" in fake_judge_model[-1]
    assert "ZEBRA-GOLD" in fake_judge_model[-1]


class JudgedTask(vf.Task[QAData]):
    @vf.reward
    async def own(self, trace) -> float:
        return 0.25


class JudgedConfig(vf.TasksetConfig):
    pass


class JudgedTaskset(vf.Taskset[JudgedTask, JudgedConfig]):
    def load(self) -> list[JudgedTask]:
        return []


async def test_task_score_runs_plugged_judges(tmp_path, fake_judge_model):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = JudgedConfig.model_validate(
        {
            "task": {
                "judges": [
                    {"id": "reference", "weight": 0.5},
                    {"id": "rubric", "path": str(rubric), "name": "quality"},
                ]
            }
        }
    )
    taskset = JudgedTaskset(cfg)
    trace = make_trace()
    await JudgedTask(trace.task.data, taskset.config.task).score(trace, runtime=None)
    assert trace.rewards["own"] == vf.Reward(score=0.25)  # decorated rewards still run
    assert trace.rewards["reference"] == vf.Reward(
        score=1.0, weight=0.5
    )  # raw score + weight, under the id-derived name
    assert (
        trace.rewards["quality"].score == 0.75
    )  # the rubric's aggregate, under its `name`
    assert (
        len(trace.info["judge_calls"]) == 2
    )  # every judge call recorded (rubric = one call)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["valid", "malformed", "wrong_subject", "out_of_range", "transport", "cancel"],
)
async def test_chat_assessor_retains_exchange_before_result_validation(
    monkeypatch, mode
):
    from verifiers.v1.assessment_runtime import execute_assessment

    source = vf.SourceSnapshot.capture(
        {"hidden_future": "DO-NOT-SEND"}, episode_id="episode", trace_ids=("trace",)
    )
    subject = vf.SubjectRef(
        kind="trace",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id="trace",
    )
    signal = vf.SignalDefinition(
        signal_id="probability",
        revision="v1",
        semantics="probability",
        description="Probability of success",
        units="probability",
        minimum=0,
        maximum=1,
    )
    view = vf.ObservationView.capture(
        {"observation": "allowed"},
        snapshot_id=source.snapshot_id,
        builder_revision="v1",
        scope="prefix",
        subjects=(subject,),
    )
    run = vf.AssessmentRun(
        run_id="run",
        producer_id="chat",
        producer_revision="v1",
        rubric_revision="v1",
        snapshot_id=source.snapshot_id,
        invocation_id="invocation",
        attempt_id="attempt",
        expected=(vf.AssessmentTarget(subject=subject, signal=signal),),
    )
    request = vf.AssessmentRequest(source=source.identity, run=run, views=(view,))
    calls = []
    raw = (
        "malformed-response"
        if mode == "malformed"
        else json.dumps({"value": 2 if mode == "out_of_range" else 0.62})
    )

    class Client:
        async def __aenter__(self):
            return SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
            )

        async def __aexit__(self, *args):
            return False

        async def create(self, **kwargs):
            calls.append(kwargs)
            if mode == "transport":
                raise RuntimeError("transport unavailable")
            if mode == "cancel":
                raise asyncio.CancelledError
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=raw))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
            )

    monkeypatch.setattr(
        "verifiers.v1.judge.build_async_openai", lambda config: Client()
    )

    def parse(response, request):
        target = request.run.expected[0]
        target_subject = target.subject
        if mode == "wrong_subject":
            target_subject = target_subject.model_copy(update={"trace_id": "wrong"})
        yield vf.Assessment(
            assessment_id="result",
            run_id=request.run.run_id,
            subject=target_subject,
            view_id=request.views[0].view_id,
            signal=target.signal,
            status="valid",
            value=json.loads(response.text)["value"],
        )

    assessor = vf.ChatAssessor(
        vf.Judge(vf.JudgeConfig(model="fixture-model")),
        lambda request, ctx: json.dumps(ctx.input(request.views[0].view_id)),
        parse,
    )
    retained = []
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await execute_assessment(assessor, request, source, retained)
    else:
        await execute_assessment(assessor, request, source, retained)
    assert len(calls) == 1
    assert "DO-NOT-SEND" not in json.dumps(calls)
    terminal = retained[-1]
    receipts = terminal.run.execution_evidence
    assert receipts[0].kind == "chat_request"
    recorded = json.loads(receipts[0].payload_json)
    assert recorded["model"] == calls[0]["model"]
    assert recorded["messages"] == calls[0]["messages"]
    if mode in {"transport", "cancel"}:
        assert receipts[-1].kind == (
            "chat_interrupted" if mode == "cancel" else "chat_failed"
        )
        assert json.loads(receipts[-1].payload_json)["usage_known"] is False
    else:
        response = json.loads(receipts[-1].payload_json)
        assert response["text"] == raw
        assert response["usage"]["completion_tokens"] == 3
    assert terminal.run.status == (
        "complete"
        if mode == "valid"
        else "interrupted"
        if mode == "cancel"
        else "failed"
    )
    if mode == "valid":
        assert terminal.assessments[0].signal.semantics == "probability"
        assert terminal.assessments[0].value == 0.62
    else:
        assert terminal.assessments == ()
    restored = vf.AssessmentBatch.model_validate_json(terminal.model_dump_json())
    assert restored == terminal
