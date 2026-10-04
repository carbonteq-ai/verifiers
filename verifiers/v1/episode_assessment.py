"""Capture episode-wide source evidence without mutable scoring outputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from verifiers.v1.assessment_source import capture_trace_source
from verifiers.v1.assessments import SourceSnapshot

if TYPE_CHECKING:
    from verifiers.v1.episode import Episode


def capture_episode_source(
    episode: Episode,
    *,
    task_evidence: Any = None,
    finalization_state: str,
) -> SourceSnapshot:
    """Retain all child inputs, including a zero-child failed execution.

    Error types describe execution state without copying exception text or appended
    assessment diagnostics into the immutable source. Child identities must already
    belong to this episode; capture never repairs contradictory ownership.
    """
    ids = [trace.id for trace in episode.traces]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate child trace identity")
    if any(trace.episode_id != episode.id for trace in episode.traces):
        raise ValueError("child trace belongs to a different episode")
    sources = [capture_trace_source(trace) for trace in episode.traces]
    return SourceSnapshot.capture(
        {
            "episode_id": episode.id,
            "task": episode.task.model_dump(mode="json"),
            "task_evidence": task_evidence,
            "execution": {
                "ok": episode.ok,
                "error_types": [error.type for error in episode.errors],
                "finalization_state": finalization_state,
            },
            "traces": [json.loads(source.source_json) for source in sources],
        },
        episode_id=episode.id,
        trace_ids=tuple(ids),
        nodes=tuple(node for source in sources for node in source.nodes),
        executions=tuple(ref for source in sources for ref in source.executions),
    )
