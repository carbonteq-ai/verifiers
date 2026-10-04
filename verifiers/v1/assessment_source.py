"""Explicit native source capture, independent of scoring and consumer tensors."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from verifiers.v1.assessments import (
    ExecutionRef,
    NodeRef,
    ObservationView,
    SourceSnapshot,
    SubjectRef,
    check_execution_scope,
    content_digest,
)

if TYPE_CHECKING:
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.trace import Trace


def _execution_prefixes(
    source: Any,
) -> dict[
    tuple[Literal["harness", "interceptor", "tool_server"], str], list[dict[str, Any]]
]:
    """Validate native receipt payloads; never correlate by arguments/provider IDs."""
    from verifiers.v1.trace import ToolExecutionEvent, ToolServerExecutionEvent

    if not isinstance(source, dict):
        raise TypeError("execution source requires a retained trace object")
    events = source.get("tool_execution_events", [])
    if not isinstance(events, list):
        raise TypeError("execution source requires a native event list")
    grouped: dict[
        tuple[Literal["harness", "interceptor", "tool_server"], str],
        list[dict[str, Any]],
    ] = {}
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise TypeError("invalid execution event")
        server = raw.get("source") == "tool_server"
        event = (
            ToolServerExecutionEvent if server else ToolExecutionEvent
        ).model_validate(raw)
        if event.receipt_seq != index:
            raise ValueError("execution receipt order changed")
        if isinstance(event, ToolServerExecutionEvent):
            from verifiers.v1._execution_links import validate_server_parent
            from verifiers.v1.mcp.execution import ToolServerReceipt

            validate_server_parent(
                ToolServerReceipt.model_validate_json(event.receipt_json),
                events[:index],
                source.get("nodes", []),
            )
        origin = event.source
        invocation = (
            event.invocation_id
            if isinstance(event, ToolServerExecutionEvent)
            else event.execution_id
        )
        if origin != "tool_server" and any(
            key[1] == invocation and key[0] not in {origin, "tool_server"}
            for key in grouped
        ):
            raise ValueError(
                "native harness/interceptor invocation origin collision is unsupported"
            )
        prior = grouped.setdefault((origin, invocation), [])
        if event.event_index != len(prior):
            raise ValueError("execution lifecycle missing or conflicting event")
        phases = [item["phase"] for item in prior] + [event.phase]
        allowed = (
            [
                ("dispatch",),
                ("dispatch", "returned"),
                ("dispatch", "raised"),
                ("dispatch", "interrupted"),
            ]
            if server
            else [
                ("before",),
                ("before", "dispatch"),
                ("before", "rejected"),
                ("before", "dispatch", "after"),
                ("before", "dispatch", "raised"),
                ("before", "dispatch", "interrupted"),
            ]
        )
        if tuple(phases) not in allowed:
            raise ValueError("invalid execution lifecycle")
        if prior and isinstance(event, ToolServerExecutionEvent):
            first = json.loads(prior[0]["receipt_json"])
            current = json.loads(event.receipt_json)
            if any(
                first.get(key) != current.get(key)
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
                raise ValueError("execution invocation contents changed")
        elif prior and isinstance(event, ToolExecutionEvent):
            if (
                prior[-1]["phase"] == "before"
                and json.loads(prior[-1]["decision_json"]).get("action") != "allow"
            ):
                raise ValueError(
                    "nonallowed harness decision cannot continue execution"
                )
            if any(
                prior[0][key] != getattr(event, key)
                for key in (
                    "node_index",
                    "emitted_call_index",
                    "generated_attempt_index",
                )
            ):
                raise ValueError("execution harness coordinates changed")
            first = json.loads(prior[0]["request_json"])
            current = json.loads(event.request_json)
            if first["call"] != current["call"]:
                raise ValueError("execution harness call changed")
        prior.append(event.model_dump(mode="json"))
    return grouped


def execution_refs(
    source: Any, *, episode_id: str, trace_id: str
) -> tuple[ExecutionRef, ...]:
    """Index the latest observed prefix for each exact native occurrence."""
    if source.get("trace_id") != trace_id:
        raise ValueError("execution trace identity differs from retained source")
    return tuple(
        ExecutionRef(
            episode_id=episode_id,
            trace_id=trace_id,
            origin=origin,
            invocation_id=invocation,
            prefix_digest=content_digest(events),
            event_count=len(events),
            phase=events[-1]["phase"],
        )
        for (origin, invocation), events in _execution_prefixes(source).items()
    )


def resolve_execution(
    source: SourceSnapshot, ref: ExecutionRef
) -> tuple[dict[str, Any], ...]:
    """Return working copies of the exact sealed lifecycle prefix, including failures."""
    payload = _execution_trace_payload(
        json.loads(source.source_json), ref.trace_id, source.episode_id
    )
    return _resolve_execution(source, ref, payload, _execution_prefixes(payload))


def resolve_execution_parent(
    source: SourceSnapshot, ref: ExecutionRef
) -> ExecutionRef | None:
    """Resolve a host-authorized server invocation to its sampled dispatch prefix.

    Historical unlinked receipts return None. This establishes a call relation;
    generated-token support must still be qualified separately by the projector.
    """
    from verifiers.v1._execution_links import validate_server_parent
    from verifiers.v1.mcp.execution import ToolServerReceipt

    payload = _execution_trace_payload(
        json.loads(source.source_json), ref.trace_id, source.episode_id
    )
    grouped = _execution_prefixes(payload)
    selected = _resolve_execution(source, ref, payload, grouped)
    if ref.origin != "tool_server":
        return None
    receipt = ToolServerReceipt.model_validate_json(selected[-1]["receipt_json"])
    parent = validate_server_parent(
        receipt,
        payload["tool_execution_events"][: selected[-1]["receipt_seq"]],
        payload.get("nodes", []),
    )
    if parent is None:
        return None
    events = grouped[(parent["source"], parent["execution_id"])]
    dispatch = events[: parent["event_index"] + 1]
    result = ExecutionRef(
        episode_id=source.episode_id,
        trace_id=ref.trace_id,
        origin=parent["source"],
        invocation_id=parent["execution_id"],
        event_count=len(dispatch),
        prefix_digest=content_digest(dispatch),
        phase="dispatch",
    )
    _resolve_execution(source, result, payload, grouped)
    return result


def _execution_trace_payload(
    payload: Any, trace_id: str, episode_id: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("execution source must be an object")
    if "traces" in payload:
        if payload.get("episode_id") != episode_id or not isinstance(
            payload["traces"], list
        ):
            raise ValueError("execution episode identity differs from retained source")
        children = [
            child
            for child in payload["traces"]
            if isinstance(child, dict) and child.get("trace_id") == trace_id
        ]
        if len(children) != 1:
            raise ValueError(
                "execution trace must resolve to exactly one retained child"
            )
        return children[0]
    if payload.get("trace_id") != trace_id:
        raise ValueError("execution trace identity differs from retained source")
    return payload


def validate_execution_refs(source: SourceSnapshot) -> None:
    """Validate the manifest with a single parse of the retained execution ledger."""
    payload = json.loads(source.source_json)
    retained = {}
    for ref in source.executions:
        if ref.trace_id not in retained:
            child = _execution_trace_payload(payload, ref.trace_id, source.episode_id)
            retained[ref.trace_id] = (child, _execution_prefixes(child))
        child, grouped = retained[ref.trace_id]
        _resolve_execution(source, ref, child, grouped)


def _resolve_execution(
    source: SourceSnapshot,
    ref: ExecutionRef,
    payload: dict[str, Any],
    grouped: dict[
        tuple[Literal["harness", "interceptor", "tool_server"], str],
        list[dict[str, Any]],
    ],
) -> tuple[dict[str, Any], ...]:
    anchors = [
        anchor
        for anchor in source.executions
        if anchor.occurrence_id == ref.occurrence_id
    ]
    if (
        not anchors
        or ref.event_count > anchors[0].event_count
        or ref.episode_id != source.episode_id
    ):
        raise ValueError("execution prefix is not a member of this source")
    if payload.get("trace_id") != ref.trace_id:
        raise ValueError("execution trace identity differs from retained source")
    events = grouped.get((ref.origin, ref.invocation_id), [])
    prefix = events[: ref.event_count]
    if (
        len(prefix) != ref.event_count
        or content_digest(prefix) != ref.prefix_digest
        or prefix[-1]["phase"] != ref.phase
    ):
        raise ValueError("execution lifecycle prefix is absent or rewritten")
    return tuple(prefix)


def capture_execution_view(
    source: SourceSnapshot,
    subjects: tuple[SubjectRef, ...],
    *,
    scope: str,
    observed: tuple[ExecutionRef, ...] | None = None,
) -> ObservationView:
    """Only selected receipt prefixes; not a claim about model conditioning/history.

    Generic ObservationView remains caller-authored. This builder has a narrow,
    source-bound contract and does not include private task/world fields implicitly.
    """
    if scope not in {"prefix", "through_action_results", "retrospective"}:
        raise ValueError("unsupported execution view scope")
    if observed is None:
        if any(subject.execution is None for subject in subjects):
            raise ValueError("execution view requires execution subjects")
        observed = tuple(
            subject.execution for subject in subjects if subject.execution is not None
        )
    if len(observed) != len(subjects):
        raise ValueError("one observed prefix is required per execution subject")
    material = []
    for subject, ref in zip(subjects, observed, strict=True):
        if (
            subject.kind != "execution"
            or subject.snapshot_id != source.snapshot_id
            or subject.episode_id != source.episode_id
        ):
            raise ValueError("execution view requires exact source execution subjects")
        assert subject.execution is not None
        if (
            subject.execution not in source.executions
            or ref.occurrence_id != subject.execution.occurrence_id
        ):
            raise ValueError(
                "observed prefix must belong to the exact assessed occurrence"
            )
        check_execution_scope(subject.model_copy(update={"execution": ref}), scope)
        material.append(
            {
                "subject_id": subject.subject_id,
                "observed": ref.model_dump(mode="json"),
                "events": resolve_execution(source, ref),
            }
        )
    return ObservationView.capture(
        material,
        snapshot_id=source.snapshot_id,
        builder_revision="native_execution_prefix_v1",
        scope=scope,
        subjects=subjects,
    )


def source_node_record(node: MessageNode) -> dict[str, Any]:
    """Stable source representation; absent optional metadata keeps legacy digests."""
    record = node.model_dump(
        mode="json",
        include={
            "parent",
            "semantic_parents",
            "message",
            "sampled",
            "token_ids",
            "mask",
            "generated_calls",
            "generated_call_producer",
        },
    )
    if not record.get("generated_calls"):
        record.pop("generated_calls", None)
    if record.get("generated_call_producer") is None:
        record.pop("generated_call_producer", None)
    return record


def capture_trace_source(trace: Trace, *, task_evidence: Any = None) -> SourceSnapshot:
    """Seal source fields without hashing mutable scores or arbitrary trace info.

    Execution ledgers and historical per-call conditioning require their own retained
    source inputs; this capture does not infer them from the final transcript.
    """
    if trace.episode_id is None:
        raise ValueError("assessment source requires the native episode identity")
    trace.validate_tool_execution_events()
    nodes = [source_node_record(node) for node in trace.nodes]
    anchors = tuple(
        NodeRef(
            trace_id=trace.id,
            node_index=index,
            node_content_digest=content_digest(node),
        )
        for index, node in enumerate(nodes)
    )
    source = {
        "trace_id": trace.id,
        "execution": {
            "error_types": [error.type for error in trace.errors],
            "finalization_state": trace.assessment_finalization_state,
        },
        "task_evidence": task_evidence,
        "task": trace.task.model_dump(mode="json"),
        "agent": trace.agent.config.model_dump(mode="json"),
        "agent_standing": {
            "trainable": trace.agent.trainable,
            "execution_purpose": trace.agent.execution_purpose,
        },
        "nodes": nodes,
        "tools": [tool.model_dump(mode="json") for tool in trace.tools],
        "calls": [
            call.model_dump(
                mode="json", include={"node", "model", "sampling", "endpoint"}
            )
            for call in trace.calls
        ],
        "artifacts": {
            path: hashlib.sha256(data).hexdigest() if data is not None else None
            for path, data in trace.state.artifacts.items()
        },
    }
    if trace.tool_execution_events:
        source["tool_execution_events"] = [
            event.model_dump(mode="json") for event in trace.tool_execution_events
        ]
    if trace.tool_state_revision:
        source["tool_state_revision"] = trace.tool_state_revision
    if trace.state_write_receipts:
        source["state_write_receipts"] = [
            write.model_dump(mode="json") for write in trace.state_write_receipts
        ]
    return SourceSnapshot.capture(
        source,
        episode_id=trace.episode_id,
        trace_ids=(trace.id,),
        nodes=anchors,
        executions=execution_refs(
            source, episode_id=trace.episode_id, trace_id=trace.id
        ),
    )
