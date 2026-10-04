"""Native domain credit assignment, independent of token alignment and advantages.

Rules select semantic recipients and declare allocation. A separate alignment
service locates original tokens; training algorithms select channels and compute
returns, advantages, normalization and losses. No assignment mutates source arrays.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import AsyncIterable, Callable, Iterable, Mapping
from typing import Any, Literal, Protocol, Self, cast

from pydantic import Field, model_validator

from verifiers.v1._validation_scope import validation_owner
from verifiers.v1.assessments import (
    Assessment,
    AssessmentRun,
    EvidenceRecord,
    SignalDefinition,
    SourceIdentity,
    SourceSnapshot,
    SubjectRef,
    canonical_json,
    content_digest,
)
from verifiers.v1.utils.decorators import invoke


class CreditRule(EvidenceRecord):
    rule_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    configuration_json: str = "{}"

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def verify(self) -> Self:
        if (
            canonical_json(json.loads(self.configuration_json))
            != self.configuration_json
        ):
            raise ValueError("assignment configuration must be canonical JSON")
        return self


class CreditTarget(EvidenceRecord):
    """Planned recipient and occurrence; a branch is optional for semantic evidence."""

    recipient: SubjectRef
    branch_id: str | None = Field(default=None, min_length=1)
    channel: str = Field(default="default", min_length=1)

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.recipient.subject_id, self.branch_id, self.channel


Allocation = Literal["turn_boundary", "broadcast", "fixed_mass"]


class CreditRequest(EvidenceRecord):
    """Accepted evidence and explicit recipients, with no inferred value mapping."""

    source: SourceSnapshot | SourceIdentity
    invocation_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    rule: CreditRule
    accepted: tuple[Assessment, ...]
    targets: tuple[CreditTarget, ...]
    allocation: Allocation
    overlap_policy: Literal["reject", "sum"]

    @property
    def request_id(self) -> str:
        return content_digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def verify(self) -> Self:
        ids = [record.assessment_id for record in self.accepted]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate accepted assessment")
        keys = [target.key for target in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate planned recipient occurrence")
        for record in self.accepted:
            _check_subject(record.subject, self.source)
        for target in self.targets:
            _check_subject(target.recipient, self.source)
        return self


class CreditGate(EvidenceRecord):
    """A rule's retained decision over accepted evidence, not a universal predicate DSL."""

    gate_id: str = Field(min_length=1)
    parent_assessment_ids: tuple[str, ...]
    outcome: Literal["passed", "blocked", "unavailable"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def verify(self) -> Self:
        if not self.parent_assessment_ids or len(self.parent_assessment_ids) != len(
            set(self.parent_assessment_ids)
        ):
            raise ValueError("gate requires unique evidence parents")
        return self


class CreditContribution(EvidenceRecord):
    """A numeric domain signal assigned to a semantic recipient, without a token mask.

    ``signal`` describes the assigned value; raw signal definitions/values remain
    on the accepted parents. ``transformation`` names the mapping under the request's
    versioned rule. Attribution is author-declared and never inferred from coordinates.
    """

    contribution_id: str = Field(min_length=1)
    parent_assessment_ids: tuple[str, ...]
    recipient: SubjectRef
    branch_id: str | None = Field(default=None, min_length=1)
    channel: str = Field(default="default", min_length=1)
    signal: SignalDefinition
    transformation: str = Field(min_length=1)
    status: Literal["valid", "unavailable", "gated"]
    value: float | None = None
    weight: float = Field(default=1.0, ge=0)
    allocation: Allocation
    attribution: Literal["exact", "coarse", "joint"] | None = None
    gates: tuple[CreditGate, ...] = ()
    reason: str | None = None

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.recipient.subject_id, self.branch_id, self.channel

    @model_validator(mode="after")
    def verify(self) -> Self:
        if not self.parent_assessment_ids or len(self.parent_assessment_ids) != len(
            set(self.parent_assessment_ids)
        ):
            raise ValueError("contribution requires unique accepted parents")
        if self.signal.semantics == "preference":
            raise ValueError("numeric assignments do not support preference signals")
        if self.status == "valid":
            if self.value is None or not math.isfinite(self.value * self.weight):
                raise ValueError("valid numeric contribution requires a finite value")
            if self.attribution is None:
                raise ValueError("valid contribution requires declared attribution")
            if self.signal.minimum is not None and self.value < self.signal.minimum:
                raise ValueError("assigned value is below its declared range")
            if self.signal.maximum is not None and self.value > self.signal.maximum:
                raise ValueError("assigned value is above its declared range")
            if any(gate.outcome != "passed" for gate in self.gates):
                raise ValueError(
                    "valid contribution cannot have blocked/unavailable gates"
                )
        elif self.value is not None or not self.reason:
            raise ValueError(
                "unavailable/gated credit has no value and requires a reason"
            )
        if self.status == "gated" and not any(
            gate.outcome == "blocked" for gate in self.gates
        ):
            raise ValueError("gated contribution requires a retained blocked gate")
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("duplicate gate decision")
        return self


class CreditAssignment(EvidenceRecord):
    schema_version: Literal[1] = 1
    request: CreditRequest
    contributions: tuple[CreditContribution, ...] = ()
    status: Literal["running", "complete", "partial", "failed", "interrupted"] = (
        "complete"
    )
    reason: str | None = None

    @property
    def assignment_id(self) -> str:
        return content_digest(self.model_dump(mode="json"))

    @property
    def missing(self) -> tuple[CreditTarget, ...]:
        returned = {record.key for record in self.contributions}
        return tuple(
            target for target in self.request.targets if target.key not in returned
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        parents = {record.assessment_id: record for record in self.request.accepted}
        targets = {target.key for target in self.request.targets}
        ids = [record.contribution_id for record in self.contributions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate contribution identity")
        returned: set[tuple[str, str | None, str]] = set()
        for contribution in self.contributions:
            _check_subject(contribution.recipient, self.request.source)
            if contribution.key not in targets:
                raise ValueError("unrequested recipient occurrence")
            if contribution.key in returned:
                raise ValueError(
                    "one contribution must aggregate each requested occurrence"
                )
            returned.add(contribution.key)
            if contribution.allocation != self.request.allocation:
                raise ValueError("contribution changed requested allocation")
            for parent_id in contribution.parent_assessment_ids:
                parent = parents.get(parent_id)
                if parent is None:
                    raise ValueError("unaccepted contribution parent")
                if contribution.status == "valid":
                    if parent.status != "valid":
                        raise ValueError("valid contribution has unavailable parent")
                    if parent.value is None:
                        raise ValueError(
                            "nonnumeric parent requires explicit derived numeric evidence"
                        )
            if (
                contribution.status == "valid"
                and contribution.transformation == "identity"
            ):
                if len(contribution.parent_assessment_ids) != 1:
                    raise ValueError(
                        "identity mapping requires exactly one numeric parent"
                    )
                parent = parents[contribution.parent_assessment_ids[0]]
                if (
                    contribution.signal != parent.signal
                    or contribution.value != parent.value
                ):
                    raise ValueError(
                        "identity mapping cannot change raw meaning or value"
                    )
            for gate in contribution.gates:
                for parent_id in gate.parent_assessment_ids:
                    parent = parents.get(parent_id)
                    if parent is None:
                        raise ValueError("unaccepted gate parent")
                    if gate.outcome != "unavailable" and parent.status != "valid":
                        raise ValueError(
                            "gate decision depends on unavailable evidence"
                        )
        if self.request.overlap_policy == "reject":
            valid = [item for item in self.contributions if item.status == "valid"]
            for index, left in enumerate(valid):
                for right in valid[index + 1 :]:
                    if left.channel != right.channel:
                        continue
                    if (
                        left.branch_id is not None
                        and right.branch_id is not None
                        and left.branch_id != right.branch_id
                    ):
                        continue
                    if _semantic_overlap(left.recipient, right.recipient):
                        raise ValueError(
                            "overlapping semantic recipients require explicit sum policy"
                        )
        if self.status == "complete" and self.missing:
            raise ValueError(
                "complete assignment must account for every planned recipient"
            )
        return self


class CreditPlanningContext(EvidenceRecord):
    """Ephemeral scoring-call boundaries for an explicitly owned credit policy.

    Planned run identities include failed current assessments; they are not a
    selection of successful findings. Prior assignments retain every status and
    snapshot. The planner decides reuse, supersession and consumed evidence;
    the runtime does not deduplicate or select the latest assessment attempt.
    """

    source: SourceIdentity
    current_assessment_runs: tuple[AssessmentRun, ...]
    prior_assignments: tuple[CreditAssignment, ...] = ()

    @model_validator(mode="after")
    def verify(self) -> Self:
        for field in ("run_id", "invocation_id", "attempt_id"):
            identities = [getattr(run, field) for run in self.current_assessment_runs]
            if len(identities) != len(set(identities)):
                raise ValueError(f"duplicate current assessment {field}")
        for run in self.current_assessment_runs:
            if run.snapshot_id != self.source.snapshot_id:
                raise ValueError("current assessment planning source mismatch")
            for target in run.expected:
                _check_subject(target.subject, self.source)
        for assignment in self.prior_assignments:
            previous = assignment.request.source
            if previous.episode_id != self.source.episode_id or not set(
                previous.trace_ids
            ) <= set(self.source.trace_ids):
                raise ValueError("prior credit planning episode or trace mismatch")
        return self


def _check_subject(
    subject: SubjectRef, source: SourceSnapshot | SourceIdentity
) -> None:
    if (
        subject.snapshot_id != source.snapshot_id
        or subject.episode_id != source.episode_id
    ):
        raise ValueError("assignment subject/source mismatch")
    if subject.trace_id is not None and subject.trace_id not in source.trace_ids:
        raise ValueError("assignment subject belongs to undeclared trace")
    if subject.execution is not None and subject.execution not in source.executions:
        raise ValueError("assignment execution prefix is absent or rewritten")
    if subject.node_index is not None:
        anchor = next(
            (
                node
                for node in source.nodes
                if node.trace_id == subject.trace_id
                and node.node_index == subject.node_index
            ),
            None,
        )
        if anchor is None or anchor.node_content_digest != subject.node_content_digest:
            raise ValueError("assignment subject node is absent or rewritten")
    for member in subject.members:
        _check_subject(member, source)


def _semantic_overlap(left: SubjectRef, right: SubjectRef) -> bool:
    """Declared semantic containment only; exact token overlap is alignment-owned."""
    if left.kind == "group":
        return any(_semantic_overlap(member, right) for member in left.members)
    if right.kind == "group":
        return any(_semantic_overlap(left, member) for member in right.members)
    if left.kind == "episode" or right.kind == "episode":
        return True
    if left.trace_id != right.trace_id:
        return False
    if left.kind == "trace" or right.kind == "trace":
        return True
    if left.kind == "execution" or right.kind == "execution":
        return (
            left.execution is not None
            and right.execution is not None
            and left.execution.occurrence_id == right.execution.occurrence_id
        )
    if left.node_index != right.node_index:
        return False
    if left.kind == "turn" or right.kind == "turn":
        return True
    if left.kind == right.kind == "call":
        return left.call_index == right.call_index
    if left.kind == right.kind == "span":
        if (left.representation, left.field, left.representation_digest) != (
            right.representation,
            right.field,
            right.representation_digest,
        ):
            return False
        assert left.span_start is not None and left.span_end is not None
        assert right.span_start is not None and right.span_end is not None
        return left.span_start < right.span_end and right.span_start < left.span_end
    return False


class AssignmentRule(Protocol):
    async def assign(self, request: CreditRequest) -> CreditAssignment: ...


@validation_owner(borrow=True)
async def execute_credit_plan(
    hooks: Mapping[str, Callable[..., Any]],
    requests: list[tuple[str, CreditRequest]],
    source: SourceSnapshot,
    retained: list[CreditAssignment],
    *,
    accepted: tuple[Assessment, ...],
) -> None:
    """Validate the entire plan before executing rules in declared request order.

    A failed rule retains its own output and does not prevent independent rules
    from running. Separate rules/channels are never aggregated here. Python rules
    ordinarily perform cheap assignment; sequential execution preserves explicit
    priority without introducing concurrency or dependency configuration.
    """
    source = SourceSnapshot.model_validate(source.model_dump(mode="python"))
    catalogue: dict[str, Assessment] = {}
    for record in accepted:
        record = Assessment.model_validate(record.model_dump(mode="python"))
        if (
            record.assessment_id in catalogue
            and catalogue[record.assessment_id] != record
        ):
            raise ValueError("retained assessment identity has conflicting contents")
        catalogue[record.assessment_id] = record
    requests = [
        (name, CreditRequest.model_validate(request.model_dump(mode="python")))
        for name, request in requests
    ]
    if {name for name, _ in requests} - set(hooks):
        raise ValueError("credit requests select unregistered hooks")
    for field in ("invocation_id", "attempt_id"):
        identities = [getattr(request, field) for _, request in requests]
        previous = {getattr(item.request, field) for item in retained}
        if len(identities) != len(set(identities)) or previous.intersection(identities):
            raise ValueError(f"duplicate assignment {field}")
    for _, request in requests:
        identity = (
            request.source.identity
            if isinstance(request.source, SourceSnapshot)
            else request.source
        )
        if identity != source.identity:
            raise ValueError("credit request source mismatch")
        if isinstance(request.source, SourceSnapshot) and request.source != source:
            raise ValueError("credit request changed retained source input")
        for record in request.accepted:
            if catalogue.get(record.assessment_id) != record:
                raise ValueError(
                    "credit request selected unretained or changed evidence"
                )
        CreditAssignment.model_validate(
            {"request": request.model_dump(mode="python"), "status": "running"}
        )

    for name, request in requests:
        # Persist the authoritative full snapshot even when planning used identity.
        request = CreditRequest.model_validate(
            request.model_dump(mode="python") | {"source": source}
        )

        def produce(request: CreditRequest, name: str = name):
            return invoke(hooks[name], {"request": request, CreditRequest: request})

        await execute_credit_assignment(produce, request, retained)


@validation_owner(borrow=True)
async def execute_credit_assignment(
    producer: AssignmentRule | Callable[..., Any],
    request: CreditRequest,
    retained: list[CreditAssignment] | None = None,
) -> CreditAssignment:
    """Validate one rule execution and retain missing/unavailable results explicitly.

    Optional native retention receives running/progress/terminal immutable records.
    A fresh explicit request selects a changed source/rule; nothing overwrites earlier
    assignments. Graph availability is neither needed nor inferred by this executor.
    """
    request = CreditRequest.model_validate(request.model_dump(mode="python"))
    if retained is not None and any(
        item.request.attempt_id == request.attempt_id for item in retained
    ):
        raise ValueError("assignment attempt has already been retained")
    # The sealed private mapping is never exposed to the rule. Reuse immutable
    # source bytes across lifecycle records rather than JSON-decoding a fresh
    # full source for every running/partial/terminal entry.
    sealed_request = request.model_dump(mode="python")
    sealed_request["source"] = request.source
    collected: list[CreditContribution] = []

    def checked(
        status: str,
        reason: str | None = None,
        contributions: tuple[CreditContribution, ...] | None = None,
    ) -> CreditAssignment:
        return CreditAssignment.model_validate(
            {
                "request": sealed_request,
                "contributions": [
                    item.model_dump(mode="python")
                    for item in (
                        tuple(collected) if contributions is None else contributions
                    )
                ],
                "status": status,
                "reason": reason,
            }
        )

    def retain(assignment: CreditAssignment) -> None:
        if retained is not None:
            retained.append(assignment)

    def accept(item: CreditContribution) -> None:
        if not isinstance(item, CreditContribution):
            raise TypeError("assignment rules must yield CreditContribution")
        candidate = checked("partial", contributions=(*collected, item))
        collected[:] = candidate.contributions
        retain(candidate)

    retain(checked("running"))
    try:
        assign = cast(Callable[..., Any], getattr(producer, "assign", producer))
        result = assign(request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, CreditAssignment):
            result = CreditAssignment.model_validate(result.model_dump(mode="python"))
            if (
                result.request != request
                or result.status == "running"
            ):
                raise ValueError(
                    "rule changed request or returned a running assignment"
                )
            candidate = checked(result.status, result.reason, result.contributions)
        else:
            if isinstance(result, AsyncIterable):
                async for item in result:
                    accept(item)
            elif isinstance(result, Iterable) and not isinstance(
                result, (str, bytes, dict)
            ):
                for item in result:
                    accept(item)
            else:
                raise TypeError("rule must return an assignment or yield contributions")
            candidate = checked("complete")
    except asyncio.CancelledError:
        retain(checked("interrupted", "cancelled"))
        raise
    except Exception as error:  # noqa: BLE001 - plugin failures become retained evidence.
        candidate = checked("failed", f"rule_error:{type(error).__name__}")
    retain(candidate)
    return candidate
