"""Project assessed subjects onto retained original sampled token coordinates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, model_validator

from verifiers.v1.assessment_source import (
    resolve_execution,
    resolve_execution_parent,
    source_node_record,
)
from verifiers.v1.assessments import (
    EvidenceRecord,
    SourceSnapshot,
    SubjectRef,
    content_digest,
)
from verifiers.v1.credit import CreditAssignment
from verifiers.v1.types import (
    AssistantMessage,
    generated_arguments_equal,
    generated_completion_digest,
    validate_generated_call_links,
    validate_generated_call_spans,
)

if TYPE_CHECKING:
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.trace import Trace


PROJECTOR_REVISION = "native-sampled-node-v1"


def token_representation_digest(node: MessageNode) -> str:
    """Identity required for an explicit span over original full-node tokens."""
    return content_digest({"token_ids": node.token_ids, "mask": node.mask})


class ProjectionInterval(EvidenceRecord):
    trace_id: str = Field(min_length=1)
    node_index: int = Field(ge=0)
    node_content_digest: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    token_digest: str = Field(min_length=1)
    sampled_mask_digest: str = Field(min_length=1)
    physical_prefix_digest: str = Field(min_length=1)
    """Physical ancestor identity, not the model call's complete conditioning input."""
    coordinate_system: Literal["node_local_full_tokens"] = "node_local_full_tokens"

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.end <= self.start:
            raise ValueError("projection interval must be nonempty and half-open")
        return self


class ProjectionResult(EvidenceRecord):
    snapshot_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    projector_revision: Literal["native-sampled-node-v1"] = PROJECTOR_REVISION
    status: Literal["exact_turn", "exact_span", "exact_call", "unsupported", "failed"]
    intervals: tuple[ProjectionInterval, ...] = ()
    reason: str | None = None
    members: tuple[ProjectionResult, ...] = ()
    """Individual results preserve group provenance even when their intervals overlap."""

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.status in {"exact_turn", "exact_span", "exact_call"}:
            if not self.intervals:
                raise ValueError("exact projection requires original sampled support")
        elif self.intervals:
            raise ValueError(
                "unavailable projection cannot expose partial policy support"
            )
        keys = [
            (item.trace_id, item.node_index, item.start, item.end)
            for item in self.intervals
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate projection interval")
        return self


class ContributionProjection(EvidenceRecord):
    contribution_id: str = Field(min_length=1)
    projection: ProjectionResult


class CreditAlignment(EvidenceRecord):
    """Original assignment plus coordinate results; no values or channels are flattened.

    Status describes coverage. A complete alignment can contain unsupported or failed
    projections; each result must be admitted separately by the selected consumer.
    Branch strings remain semantic occurrences, never inferred physical ordinals.
    """

    assignment: CreditAssignment
    contributions: tuple[ContributionProjection, ...]
    status: Literal["complete", "partial"] = "complete"

    @property
    def missing(self) -> tuple[str, ...]:
        returned = {item.contribution_id for item in self.contributions}
        return tuple(
            item.contribution_id
            for item in self.assignment.contributions
            if item.contribution_id not in returned
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        source = self.assignment.request.source
        if not isinstance(source, SourceSnapshot):
            raise ValueError(  # noqa: TRY004 - a permitted source union lacks evidence.
                "alignment requires the retained full source snapshot"
            )
        expected = {
            item.contribution_id: item for item in self.assignment.contributions
        }
        returned: set[str] = set()
        for item in self.contributions:
            contribution = expected.get(item.contribution_id)
            if contribution is None or item.contribution_id in returned:
                raise ValueError("unexpected or duplicate contribution projection")
            returned.add(item.contribution_id)
            if (
                item.projection.snapshot_id != source.snapshot_id
                or item.projection.subject_id != contribution.recipient.subject_id
            ):
                raise ValueError(
                    "projection source/recipient does not match assignment"
                )
            if contribution.status != "valid" and item.projection.status in {
                "exact_turn",
                "exact_span",
                "exact_call",
            }:
                raise ValueError(
                    "unavailable/gated contribution cannot expose policy support"
                )
        if self.status == "complete" and self.missing:
            raise ValueError("complete alignment must account for every contribution")
        return self


def project_assignment(
    assignment: CreditAssignment,
    traces: Mapping[str, Trace] | Trace,
) -> CreditAlignment:
    """Align every retained semantic contribution without changing its domain meaning."""
    assignment = CreditAssignment.model_validate(assignment.model_dump(mode="python"))
    source = assignment.request.source
    if not isinstance(source, SourceSnapshot):
        raise ValueError(  # noqa: TRY004 - a permitted source union lacks evidence.
            "assignment alignment requires the retained full source snapshot"
        )
    if not isinstance(traces, Mapping):
        traces = {traces.id: traces}
    saved = json.loads(source.source_json)
    projected: list[ContributionProjection] = []
    for contribution in assignment.contributions:
        if contribution.status == "valid":
            projection = _project_subject(contribution.recipient, source, traces, saved)
        else:
            projection = ProjectionResult(
                snapshot_id=source.snapshot_id,
                subject_id=contribution.recipient.subject_id,
                status="unsupported",
                reason=f"contribution_{contribution.status}",
            )
        projected.append(
            ContributionProjection(
                contribution_id=contribution.contribution_id,
                projection=projection,
            )
        )
    return CreditAlignment(assignment=assignment, contributions=tuple(projected))


def project_subject(
    subject: SubjectRef,
    source: SourceSnapshot,
    traces: Mapping[str, Trace] | Trace,
) -> ProjectionResult:
    """Locate original policy support without reconstructing or tokenizing any text.

    Native trace and episode snapshots retain the same child node representation.
    Calls and text spans remain unsupported until a renderer supplies verified
    original completion spans. Group results are exact unions of their retained
    member coordinates, all-or-unavailable; a failed member never disappears.
    Exact original coordinates do not establish complete model conditioning or
    causal attribution; compacted windows and other context need their own evidence.
    """
    subject = SubjectRef.model_validate(subject.model_dump(mode="python"))
    source = SourceSnapshot.model_validate(source.model_dump(mode="python"))
    if not isinstance(traces, Mapping):
        traces = {traces.id: traces}
    return _project_subject(subject, source, traces, json.loads(source.source_json))


def _project_subject(
    subject: SubjectRef,
    source: SourceSnapshot,
    traces: Mapping[str, Trace],
    saved_payload: Any,
) -> ProjectionResult:
    """Already validated immutable subject/source; inspect current nodes afresh."""

    def result(
        status: Literal[
            "exact_turn", "exact_span", "exact_call", "unsupported", "failed"
        ],
        reason: str | None = None,
        intervals: tuple[ProjectionInterval, ...] = (),
        members: tuple[ProjectionResult, ...] = (),
    ) -> ProjectionResult:
        return ProjectionResult(
            snapshot_id=source.snapshot_id,
            subject_id=subject.subject_id,
            status=status,
            reason=reason,
            intervals=intervals,
            members=members,
        )

    if (
        subject.snapshot_id != source.snapshot_id
        or subject.episode_id != source.episode_id
    ):
        return result("failed", "subject_source_mismatch")
    if subject.kind == "group":
        members = tuple(
            _project_subject(member, source, traces, saved_payload)
            for member in subject.members
        )
        if any(member.status == "failed" for member in members):
            return result("failed", "group_member_failed", members=members)
        if any(member.status == "unsupported" for member in members):
            return result("unsupported", "group_member_unsupported", members=members)
        return result("exact_span", intervals=_union(members), members=members)
    if subject.trace_id is not None and subject.trace_id not in source.trace_ids:
        return result("failed", "undeclared_trace")
    if subject.kind == "execution":
        if subject.execution not in source.executions:
            return result("failed", "execution_prefix_absent_or_rewritten")
        assert subject.execution is not None
        assert subject.trace_id is not None
        try:
            selected = resolve_execution(source, subject.execution)
            parent_ref = (
                resolve_execution_parent(source, subject.execution)
                if subject.execution.origin == "tool_server"
                else subject.execution
            )
            parent_events = (
                resolve_execution(source, parent_ref) if parent_ref is not None else ()
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            return result("failed", "execution_parent_evidence_inconsistent")
        if parent_ref is None:
            return result(
                "unsupported", "execution_generated_call_token_relation_unqualified"
            )
        dispatches = [event for event in parent_events if event["phase"] == "dispatch"]
        if len(dispatches) != 1:
            return result("unsupported", "execution_sampled_dispatch_unavailable")
        dispatch = dispatches[0]
        current = traces.get(subject.trace_id)
        if (
            current is None
            or current.id != subject.trace_id
            or current.episode_id != source.episode_id
        ):
            return result("failed", "current_trace_identity_mismatch")
        for retained in (*selected, *parent_events):
            sequence = retained["receipt_seq"]
            if sequence >= len(current.tool_execution_events):
                return result("failed", "current_tool_execution_events_changed")
            event = current.tool_execution_events[sequence]
            try:
                admitted = (
                    type(event)
                    .model_validate(event.model_dump(mode="python"), strict=True)
                    .model_dump(mode="json")
                )
            except (ValueError, TypeError, AttributeError):
                return result("failed", "current_tool_execution_events_changed")
            if admitted != retained:
                return result("failed", "current_tool_execution_events_changed")
        attempt_index = dispatch.get("generated_attempt_index")
        if attempt_index is None:
            return result(
                "unsupported", "execution_generated_call_token_relation_unqualified"
            )
        node_index = dispatch["node_index"]
        anchor = next(
            (
                node
                for node in source.nodes
                if node.trace_id == subject.trace_id and node.node_index == node_index
            ),
            None,
        )
        if anchor is None:
            return result("failed", "missing_node_anchor")
        if node_index >= len(current.nodes):
            return result("failed", "node_absent")
        node = current.nodes[node_index]
        emitted_index = dispatch["emitted_call_index"]
        if not isinstance(node.message, AssistantMessage):
            return result("failed", "execution_sampled_call_owner_changed")
        calls = node.message.tool_calls or []
        if emitted_index >= len(calls) or attempt_index >= len(node.generated_calls):
            return result("failed", "execution_generated_call_relation_changed")
        emitted = calls[emitted_index]
        request_call = json.loads(dispatch["request_json"])["call"]
        if (
            node.generated_calls[attempt_index].emitted_call_index != emitted_index
            or any(
                request_call[key] != getattr(emitted, key)
                for key in ("id", "name", "type")
            )
            or not generated_arguments_equal(
                request_call["arguments"], emitted.arguments
            )
            or sum(call.id == emitted.id for call in calls) != 1
        ):
            return result("failed", "execution_generated_call_relation_changed")
        mapped = SubjectRef(
            kind="call",
            snapshot_id=source.snapshot_id,
            episode_id=source.episode_id,
            trace_id=subject.trace_id,
            node_index=node_index,
            node_content_digest=anchor.node_content_digest,
            call_index=attempt_index,
        )
        projection = _project_subject(mapped, source, traces, saved_payload)
        return result(
            projection.status, projection.reason, projection.intervals, (projection,)
        )
    if subject.kind in {"episode", "trace"}:
        failure = _whole_source_failure(subject, source, traces, saved_payload)
        if failure is not None:
            return result("failed", failure)
        return result("unsupported", "explicit_turn_or_span_required")
    assert subject.trace_id is not None
    trace = traces.get(subject.trace_id)
    if (
        trace is None
        or trace.id != subject.trace_id
        or trace.episode_id != source.episode_id
    ):
        return result("failed", "current_trace_identity_mismatch")
    if not trace.agent.trainable:
        return result("unsupported", "non_trainable_trace")
    if trace.agent.execution_purpose == "assessment":
        return result("unsupported", "assessment_execution_trace")
    saved = saved_payload
    if not isinstance(saved, dict):
        return result("unsupported", "unsupported_source_layout")
    if "traces" in saved:
        if saved.get("episode_id") != source.episode_id:
            return result("failed", "retained_episode_identity_mismatch")
        children = saved["traces"]
        if not isinstance(children, list) or any(
            not isinstance(child, dict) or not isinstance(child.get("trace_id"), str)
            for child in children
        ):
            return result("failed", "invalid_retained_child_sources")
        child_ids = [child["trace_id"] for child in children]
        if len(child_ids) != len(set(child_ids)):
            return result("failed", "duplicate_retained_child_trace")
        if set(child_ids) != set(source.trace_ids):
            return result("failed", "retained_child_manifest_mismatch")
        matches = [child for child in children if child["trace_id"] == subject.trace_id]
        if len(matches) != 1:
            return result("failed", "missing_retained_child_trace")
        saved = matches[0]
    elif "trace_id" in saved:
        if saved["trace_id"] != subject.trace_id:
            return result("failed", "missing_retained_child_trace")
        if source.trace_ids != (subject.trace_id,):
            return result("failed", "retained_child_manifest_mismatch")
    else:
        return result("unsupported", "unsupported_source_layout")
    standing = saved.get("agent_standing")
    if standing is not None and standing != {
        "trainable": trace.agent.trainable,
        "execution_purpose": trace.agent.execution_purpose,
    }:
        return result("failed", "current_agent_standing_mismatch")
    saved_nodes = saved.get("nodes")
    if not isinstance(saved_nodes, list):
        return result("failed", "missing_retained_nodes")
    assert subject.node_index is not None
    index = subject.node_index
    if index >= len(saved_nodes) or index >= len(trace.nodes):
        return result("failed", "node_absent")
    anchors = {
        node.node_index: node for node in source.nodes if node.trace_id == trace.id
    }
    ancestor_digests: list[str] = []
    cursor: int | None = index
    visited: set[int] = set()
    while cursor is not None:
        if (
            cursor in visited
            or cursor < 0
            or cursor >= len(saved_nodes)
            or cursor >= len(trace.nodes)
        ):
            return result("failed", "invalid_conditioning_path")
        visited.add(cursor)
        anchor = anchors.get(cursor)
        if anchor is None:
            return result("failed", "missing_node_anchor")
        retained_digest = content_digest(saved_nodes[cursor])
        current_digest = content_digest(source_node_record(trace.nodes[cursor]))
        if (
            retained_digest != anchor.node_content_digest
            or current_digest != retained_digest
        ):
            return result("failed", "source_node_changed")
        ancestor_digests.append(retained_digest)
        cursor = trace.nodes[cursor].parent
    if subject.node_content_digest != anchors[index].node_content_digest:
        return result("failed", "subject_node_digest_mismatch")
    node = trace.nodes[index]
    if node.message.role != "assistant" or node.sampled is not True:
        return result("unsupported", "not_sampled_assistant")
    if not node.token_ids:
        return result("unsupported", "original_tokens_not_available")
    if len(node.mask) != len(node.token_ids):
        return result("failed", "sampled_mask_length_mismatch")
    if any(type(value) is not bool for value in node.mask):
        return result("failed", "sampled_mask_type_mismatch")
    if any(type(value) is not int or value < 0 for value in node.token_ids):
        return result("failed", "original_token_type_mismatch")
    if subject.kind == "call":
        try:
            validate_generated_call_links(
                node.generated_calls, node.message.tool_calls or ()
            )
            validate_generated_call_spans(node.generated_calls)
        except (ValueError, TypeError, AttributeError):
            return result("failed", "generated_call_evidence_inconsistent")
        assert subject.call_index is not None
        if subject.call_index >= len(node.generated_calls):
            return result("unsupported", "original_call_spans_not_available")
        attempt = node.generated_calls[subject.call_index]
        if attempt.attempt_index != subject.call_index:
            return result("failed", "generated_call_ordinal_mismatch")
        if attempt.span_fidelity != "exact" or attempt.token_span is None:
            return result("unsupported", "generated_call_span_joint_or_unavailable")
        if attempt.parser_revision.startswith("unqualified:"):
            return result("unsupported", "generated_call_parser_provenance_unqualified")
        if attempt.coordinate_system != "node_local_full_tokens":
            return result("failed", "generated_call_coordinate_mismatch")
        if attempt.completion_token_digest != generated_completion_digest(
            token
            for token, eligible in zip(node.token_ids, node.mask, strict=True)
            if eligible
        ):
            return result("failed", "generated_call_completion_digest_mismatch")
        start, end = attempt.token_span
        if end > len(node.token_ids) or not all(node.mask[start:end]):
            return result("failed", "generated_call_span_not_sampled")
        spans = ((start, end),)
        status = "exact_call"
    elif subject.kind == "span":
        if subject.representation != "full_tokens":
            return result("unsupported", "text_span_has_no_verified_token_alignment")
        if subject.representation_digest != token_representation_digest(node):
            return result("failed", "token_representation_digest_mismatch")
        assert subject.span_start is not None and subject.span_end is not None
        start, end = subject.span_start, subject.span_end
        if end > len(node.token_ids):
            return result("failed", "span_out_of_bounds")
        if not all(node.mask[start:end]):
            return result("failed", "span_contains_unsampled_tokens")
        spans = ((start, end),)
        status = "exact_span"
    else:
        spans = _sampled_spans(node.mask)
        status = "exact_turn"
        if not spans:
            return result("unsupported", "no_sampled_policy_tokens")
    intervals = tuple(
        ProjectionInterval(
            trace_id=trace.id,
            node_index=index,
            node_content_digest=anchors[index].node_content_digest,
            start=start,
            end=end,
            token_digest=content_digest(node.token_ids),
            sampled_mask_digest=content_digest(node.mask),
            physical_prefix_digest=content_digest(tuple(reversed(ancestor_digests))),
        )
        for start, end in spans
    )
    return result(status, intervals=intervals)


def _whole_source_failure(
    subject: SubjectRef,
    source: SourceSnapshot,
    traces: Mapping[str, Trace],
    saved: Any,
) -> str | None:
    """Verify whole-outcome ownership and retained transcript without requiring tokens.

    Whole-episode findings describe the sealed execution, so appending another
    action changes their source. Mutable rewards and annotation fields are ignored.
    External task evidence remains the immutable snapshot's authority.
    """
    if not isinstance(saved, dict):
        return "unsupported_source_layout"
    if "traces" in saved:
        children = saved["traces"]
        if saved.get("episode_id") != source.episode_id or not isinstance(
            children, list
        ):
            return "retained_episode_identity_mismatch"
    elif "trace_id" in saved:
        children = [saved]
    else:
        return "unsupported_source_layout"
    if any(
        not isinstance(child, dict) or not isinstance(child.get("trace_id"), str)
        for child in children
    ):
        return "invalid_retained_child_sources"
    identities = [child["trace_id"] for child in children]
    if len(set(identities)) != len(identities) or set(identities) != set(
        source.trace_ids
    ):
        return "retained_child_manifest_mismatch"
    selected = (
        children
        if subject.kind == "episode"
        else [child for child in children if child["trace_id"] == subject.trace_id]
    )
    if subject.kind == "episode" and {
        trace.id for trace in traces.values() if trace.episode_id == source.episode_id
    } != set(source.trace_ids):
        return "current_episode_child_manifest_mismatch"
    if subject.kind == "trace" and len(selected) != 1:
        return "missing_retained_child_trace"
    anchors = {
        (node.trace_id, node.node_index): node.node_content_digest
        for node in source.nodes
    }
    for child in selected:
        current = traces.get(child["trace_id"])
        if (
            current is None
            or current.id != child["trace_id"]
            or current.episode_id != source.episode_id
        ):
            return "current_trace_identity_mismatch"
        if child.get("task") != current.task.model_dump(mode="json") or child.get(
            "agent"
        ) != current.agent.config.model_dump(mode="json"):
            return "current_execution_configuration_mismatch"
        if child.get("agent_standing") != {
            "trainable": current.agent.trainable,
            "execution_purpose": current.agent.execution_purpose,
        }:
            return "current_agent_standing_mismatch"
        if child.get("tools") != [
            tool.model_dump(mode="json") for tool in current.tools
        ]:
            return "current_tools_changed"
        if child.get("calls") != [
            call.model_dump(
                mode="json",
                include={"node", "model", "sampling", "endpoint"},
            )
            for call in current.calls
        ]:
            return "current_calls_changed"
        if child.get("tool_execution_events", []) != [
            event.model_dump(mode="json") for event in current.tool_execution_events
        ]:
            return "current_tool_execution_events_changed"
        if child.get("tool_state_revision", 0) != current.tool_state_revision:
            return "current_tool_state_revision_changed"
        if child.get("state_write_receipts", []) != [
            write.model_dump(mode="json") for write in current.state_write_receipts
        ]:
            return "current_state_write_receipts_changed"
        if child.get("artifacts") != {
            path: hashlib.sha256(data).hexdigest() if data is not None else None
            for path, data in current.state.artifacts.items()
        }:
            return "current_artifacts_changed_or_unavailable"
        nodes = child.get("nodes")
        if not isinstance(nodes, list) or len(nodes) != len(current.nodes):
            return "whole_source_node_count_changed"
        for index, (retained, node) in enumerate(
            zip(nodes, current.nodes, strict=True)
        ):
            digest = content_digest(retained)
            if digest != anchors.get((current.id, index)) or digest != content_digest(
                source_node_record(node)
            ):
                return "source_node_changed"
    return None


def _sampled_spans(mask: list[bool]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, sampled in enumerate((*mask, False)):
        if sampled and start is None:
            start = index
        elif not sampled and start is not None:
            spans.append((start, index))
            start = None
    return tuple(spans)


def _union(members: tuple[ProjectionResult, ...]) -> tuple[ProjectionInterval, ...]:
    selected: list[ProjectionInterval] = []
    intervals = sorted(
        (interval for member in members for interval in member.intervals),
        key=lambda interval: (
            interval.trace_id,
            interval.node_index,
            interval.start,
            interval.end,
        ),
    )
    for interval in intervals:
        if selected:
            previous = selected[-1]
            if (
                previous.trace_id == interval.trace_id
                and previous.node_index == interval.node_index
                and interval.start <= previous.end
            ):
                selected[-1] = previous.model_copy(
                    update={"end": max(previous.end, interval.end)}
                )
                continue
        selected.append(interval)
    return tuple(selected)
