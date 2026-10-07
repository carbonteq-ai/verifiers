"""Immutable assessment evidence; policy credit and token projection live elsewhere.

Retained JSON is canonical text rather than mutable dictionaries. A frozen Pydantic
model alone does not prevent mutation of a nested dictionary or list.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import AsyncIterable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from verifiers.v1._validation_scope import intrinsic_proof, proven_instance


def canonical_json(value: Any) -> str:
    """Encode declared JSON evidence without nonfinite numbers or implicit coercion."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical(text: str) -> str:
    return canonical_json(json.loads(text))


ARCHIVE_SAME = "__verifiers_archive_same__"
"""Key of an in-process placeholder for an already serialized source or view."""


@dataclass
class _ArchiveScope:
    token: str
    seen: dict[int, Any] = field(default_factory=dict)
    emitted: int = 0


_archive_scope: ContextVar[_ArchiveScope | None] = ContextVar(
    "assessment_archive_scope", default=None
)


@contextmanager
def archive_serialization() -> Iterator[_ArchiveScope]:
    """Scope one archive encoding; placeholders carry the scope's token.

    Within the scope, a source or view object that a batch or credit request
    already serialized is emitted again only as a placeholder naming the token
    and its identity. ``normalize_history`` called with the same token resolves
    placeholders to that identity's pooled entry; foreign or stale tokens are
    never honored. Distinct objects, even with equal content, are serialized in
    full and compared as before.
    """
    scope = _ArchiveScope(secrets.token_hex(16))
    token = _archive_scope.set(scope)
    try:
        yield scope
    finally:
        _archive_scope.reset(token)
        scope.seen.clear()


def _placeholder(scope: _ArchiveScope, model: Any) -> dict[str, Any] | None:
    if isinstance(model, SourceSnapshot):
        kind, identity = "source", model.snapshot_id
    elif isinstance(model, ObservationView):
        kind, identity = "view", model.view_id
    else:
        return None
    if scope.seen.get(id(model)) is not model:
        return None
    scope.emitted += 1
    return {ARCHIVE_SAME: [scope.token, kind, identity]}


def _archive_field(value: Any, handler: Any, info: Any) -> Any:
    """Serialize a source or views field, eliding objects already emitted."""
    scope = _archive_scope.get()
    # Field filters apply per position: filtered fields are serialized normally.
    if scope is None or info.include is not None or info.exclude is not None:
        return handler(value)
    if isinstance(value, tuple):
        marks = [_placeholder(scope, item) for item in value]
        if marks and all(mark is not None for mark in marks):
            return marks
    else:
        mark = _placeholder(scope, value)
        if mark is not None:
            return mark
    data = handler(value)
    # The scope holds each object, so its id stays unique until the scope ends.
    for item in value if isinstance(value, tuple) else (value,):
        if isinstance(item, (SourceSnapshot, ObservationView)):
            scope.seen[id(item)] = item
    return data


def _membership(items: tuple[Any, ...]):
    """``item in items`` through a hash set when every value hashes.

    Frozen records hash their field values consistently with ``==``, so the set
    answers exactly as the tuple scan would; unhashable (forged) values fall
    back to the scan.
    """
    try:
        index = frozenset(items)
    except TypeError:
        return items.__contains__

    def contains(item: Any) -> bool:
        try:
            return item in index
        except TypeError:
            return item in items

    return contains


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ArtifactRef(EvidenceRecord):
    """A retained artifact whose bytes are verified by its owning artifact transport."""

    uri: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    media_type: str = "application/json"


class NodeRef(EvidenceRecord):
    trace_id: str = Field(min_length=1)
    node_index: int = Field(ge=0, strict=True)
    node_content_digest: str = Field(min_length=1)


class ExecutionRef(EvidenceRecord):
    """An occurrence identity plus an exact retained lifecycle visibility boundary.

    These are coordinates only. Arguments/results stay in the sealed source, never
    in producer-facing SourceIdentity. A later return changes prefix_digest, not
    occurrence_id. Neither identity claims generated-call or token attribution.
    """

    episode_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    origin: Literal["harness", "interceptor", "tool_server"]
    invocation_id: str = Field(min_length=1)
    prefix_digest: str = Field(min_length=1)
    event_count: int = Field(ge=1, strict=True)
    phase: Literal[
        "before", "dispatch", "after", "returned", "raised", "rejected", "interrupted"
    ]

    @property
    def occurrence_id(self) -> str:
        return _occurrence_id(
            self.episode_id, self.trace_id, self.origin, self.invocation_id
        )


@lru_cache(maxsize=65536, typed=True)
def _occurrence_id(
    episode_id: Any, trace_id: Any, origin: Any, invocation_id: Any
) -> str:
    # Source identities recompute every execution's occurrence on each
    # validation. typed=True keeps True, 1 and 1.0 as separate entries, so a
    # copied coordinate type still gets its own digest.
    return content_digest(
        {
            "episode_id": episode_id,
            "trace_id": trace_id,
            "origin": origin,
            "invocation_id": invocation_id,
        }
    )


class SourceIdentity(EvidenceRecord):
    """Source coordinates without any policy-hidden or future input payload."""

    snapshot_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    nodes: tuple[NodeRef, ...] = ()
    trace_ids: tuple[str, ...] = ()
    executions: tuple[ExecutionRef, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def verify_nodes(self) -> Self:
        keys = [(node.trace_id, node.node_index) for node in self.nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate source node")
        if len(self.trace_ids) != len(set(self.trace_ids)) or any(
            not trace_id for trace_id in self.trace_ids
        ):
            raise ValueError("trace identities must be nonempty and unique")
        if any(node.trace_id not in self.trace_ids for node in self.nodes):
            raise ValueError("source node belongs to an undeclared trace")
        occurrence_ids = [ref.occurrence_id for ref in self.executions]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("duplicate or conflicting execution occurrence")
        if any(
            ref.episode_id != self.episode_id or ref.trace_id not in self.trace_ids
            for ref in self.executions
        ):
            raise ValueError("execution belongs to another source episode/trace")
        return self


class SourceSnapshot(EvidenceRecord):
    """An explicitly selected source revision, independent of appended assessments.

    The caller selects source fields before capture. This layer never recursively
    deletes field names: tool arguments or task data may legitimately use those names.
    """

    schema_version: Literal[1, 2] = 1
    snapshot_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    source_json: str
    source_digest: str
    nodes: tuple[NodeRef, ...] = ()
    trace_ids: tuple[str, ...] = ()
    executions: tuple[ExecutionRef, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("source schema version requires an exact integer")
        return value

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            snapshot_id=self.snapshot_id,
            episode_id=self.episode_id,
            nodes=self.nodes,
            trace_ids=self.trace_ids,
            executions=self.executions,
        )

    @classmethod
    def capture(
        cls,
        source: Any,
        *,
        episode_id: str,
        nodes: tuple[NodeRef, ...] = (),
        trace_ids: tuple[str, ...] | None = None,
        executions: tuple[ExecutionRef, ...] = (),
    ) -> Self:
        if trace_ids is None:
            trace_ids = tuple(dict.fromkeys(node.trace_id for node in nodes))
        digest = content_digest(source)
        identity = {
            "episode_id": episode_id,
            "source": digest,
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "trace_ids": trace_ids,
        }
        if executions:
            identity["executions"] = [ref.model_dump(mode="json") for ref in executions]
        return cls(
            schema_version=2 if executions else 1,
            snapshot_id=content_digest(identity),
            episode_id=episode_id,
            source_json=canonical_json(source),
            source_digest=digest,
            nodes=nodes,
            trace_ids=trace_ids,
            executions=executions,
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        if proven_instance(self):
            return self
        if type(self.schema_version) is not int:
            raise ValueError("source schema version requires an exact integer")
        SourceIdentity.model_validate(
            self.model_dump(
                mode="python",
                exclude={"schema_version", "source_json", "source_digest"},
            ),
            strict=True,
        )
        proof = intrinsic_proof(self)
        if proof.hit:
            return self
        if self.source_json != _canonical(self.source_json):
            raise ValueError("source_json must be canonical JSON")
        source = json.loads(self.source_json)
        if content_digest(source) != self.source_digest:
            raise ValueError("source digest does not match retained input")
        identity = {
            "episode_id": self.episode_id,
            "source": self.source_digest,
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
            "trace_ids": self.trace_ids,
        }
        if self.executions:
            identity["executions"] = [
                ref.model_dump(mode="json") for ref in self.executions
            ]
        if self.schema_version != (2 if self.executions else 1):
            raise ValueError("execution source requires schema version two")
        expected = content_digest(identity)
        if self.snapshot_id != expected:
            raise ValueError("snapshot identity does not match source")
        _ = self.identity
        if self.executions:
            from verifiers.v1.assessment_source import validate_execution_refs

            validate_execution_refs(self)
        proof.remember()
        return self


SubjectKind = Literal["episode", "trace", "turn", "call", "span", "group", "execution"]


class SubjectRef(EvidenceRecord):
    """Stable subject against a snapshot; call occurrence is node-local, not provider ID."""

    kind: SubjectKind
    snapshot_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    trace_id: str | None = None
    node_index: int | None = Field(default=None, ge=0, strict=True)
    node_content_digest: str | None = None
    call_index: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Original generated-attempt ordinal within the sampled node; not an execution occurrence or model-call index",
    )
    span_start: int | None = Field(default=None, ge=0, strict=True)
    span_end: int | None = Field(default=None, ge=0, strict=True)
    representation: (
        Literal["full_tokens", "utf8_bytes", "unicode_codepoints"] | None
    ) = None
    field: str | None = None
    representation_digest: str | None = None
    members: tuple[SubjectRef, ...] = ()
    execution: ExecutionRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def subject_id(self) -> str:
        payload = self.model_dump(mode="json")
        if self.execution is None:
            payload.pop("execution", None)
        return content_digest(payload)

    @model_validator(mode="after")
    def verify_shape(self) -> Self:
        node_fields = (self.node_index, self.node_content_digest)
        span_fields = (
            self.span_start,
            self.span_end,
            self.representation,
            self.representation_digest,
        )
        if self.kind == "episode" and self.trace_id is not None:
            raise ValueError("episode subject cannot name a trace")
        if (
            self.kind in {"trace", "turn", "call", "span", "execution"}
            and not self.trace_id
        ):
            raise ValueError("trace-local subject requires trace_id")
        if (self.kind == "execution") != (self.execution is not None):
            raise ValueError("only execution subjects require an execution reference")
        if self.execution is not None and (
            self.execution.episode_id != self.episode_id
            or self.execution.trace_id != self.trace_id
        ):
            raise ValueError("execution subject coordinates disagree")
        if self.kind in {"turn", "call", "span"}:
            if any(value is None for value in node_fields):
                raise ValueError("node-local subject requires index and content digest")
        elif any(value is not None for value in node_fields):
            raise ValueError("non-node subject cannot name a node")
        if (self.kind == "call") != (self.call_index is not None):
            raise ValueError("only call subjects require call_index")
        if self.kind == "span":
            if any(value is None for value in span_fields):
                raise ValueError(
                    "span requires coordinates and representation identity"
                )
            if self.span_start >= self.span_end:  # type: ignore[operator]
                raise ValueError("span must be a nonempty half-open interval")
            if self.representation != "full_tokens" and not self.field:
                raise ValueError("text span requires an exact content field")
        elif any(value is not None for value in (*span_fields, self.field)):
            raise ValueError("only span subjects can carry span coordinates")
        if self.kind == "group":
            if self.trace_id is not None or not self.members:
                raise ValueError("group requires members and no trace_id")
            ids = [member.subject_id for member in self.members]
            if len(ids) != len(set(ids)):
                raise ValueError("group has duplicate members")
            if any(
                member.snapshot_id != self.snapshot_id
                or member.episode_id != self.episode_id
                for member in self.members
            ):
                raise ValueError("group members must share the source episode/snapshot")
        elif self.members:
            raise ValueError("only groups carry members")
        return self


def check_execution_scope(subject: SubjectRef, scope: str) -> None:
    """Check selected receipt visibility, not arbitrary caller-authored context."""
    if subject.execution is not None:
        terminal = subject.execution.phase in {
            "after",
            "returned",
            "raised",
            "rejected",
            "interrupted",
        }
        if scope == "prefix" and terminal:
            raise ValueError(
                "prefix execution context cannot include terminal receipts"
            )
        if scope == "through_action_results" and not terminal:
            raise ValueError("execution result context requires a terminal receipt")
    for member in subject.members:
        check_execution_scope(member, scope)


class ObservationView(EvidenceRecord):
    view_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    builder_revision: str = Field(min_length=1)
    scope: Literal["prefix", "through_action_results", "retrospective"]
    subjects: tuple[SubjectRef, ...]
    input_json: str | None = None
    artifact: ArtifactRef | None = None
    input_digest: str = Field(min_length=1)

    @classmethod
    def capture(
        cls,
        value: Any,
        *,
        snapshot_id: str,
        builder_revision: str,
        scope: Literal["prefix", "through_action_results", "retrospective"],
        subjects: tuple[SubjectRef, ...],
    ) -> Self:
        fields = {
            "snapshot_id": snapshot_id,
            "builder_revision": builder_revision,
            "scope": scope,
            "subjects": subjects,
            "input_json": canonical_json(value),
            "input_digest": content_digest(value),
        }
        identity = dict(fields, subjects=[s.model_dump(mode="json") for s in subjects])
        return cls(view_id=content_digest(identity), **fields)

    @model_validator(mode="after")
    def verify(self) -> Self:
        if proven_instance(self):
            return self
        if (
            type(self.subjects) is not tuple
            or self.scope not in {"prefix", "through_action_results", "retrospective"}
            or any(
                type(value) is not str or not value
                for value in (
                    self.view_id,
                    self.snapshot_id,
                    self.builder_revision,
                    self.input_digest,
                )
            )
            or self.input_json is not None
            and type(self.input_json) is not str
        ):
            raise ValueError("view metadata requires exact supported types")
        for subject in self.subjects:
            SubjectRef.model_validate(subject.model_dump(mode="python"), strict=True)
        if self.artifact is not None:
            ArtifactRef.model_validate(
                self.artifact.model_dump(mode="python"), strict=True
            )
        proof = intrinsic_proof(self)
        if proof.hit:
            return self
        if (self.input_json is None) == (self.artifact is None):
            raise ValueError("retain exactly one input JSON or immutable artifact")
        if (
            self.builder_revision == "native_execution_prefix_v1"
            and self.input_json is None
        ):
            raise ValueError(
                "execution prefix view requires inspectable canonical input"
            )
        if self.input_json is not None:
            if self.input_json != _canonical(self.input_json):
                raise ValueError("input_json must be canonical JSON")
            if content_digest(json.loads(self.input_json)) != self.input_digest:
                raise ValueError("view input digest mismatch")
            if self.builder_revision == "native_execution_prefix_v1":
                material = json.loads(self.input_json)
                if not isinstance(material, list) or len(material) != len(
                    self.subjects
                ):
                    raise ValueError(
                        "execution view must retain one prefix per subject"
                    )
                for subject, item in zip(self.subjects, material, strict=True):
                    if not isinstance(item, dict):
                        raise TypeError("execution view prefix must be an object")
                    item = cast(dict[str, Any], item)
                    if set(item) != {"subject_id", "observed", "events"}:
                        raise ValueError("execution view has undeclared context")
                    observed = ExecutionRef.model_validate(item["observed"])
                    if (
                        subject.execution is None
                        or observed.occurrence_id != subject.execution.occurrence_id
                        or item["subject_id"] != subject.subject_id
                    ):
                        raise ValueError("execution view subject/prefix mismatch")
                    check_execution_scope(
                        subject.model_copy(update={"execution": observed}), self.scope
                    )
                    if (
                        not isinstance(item["events"], list)
                        or not item["events"]
                        or any(not isinstance(event, dict) for event in item["events"])
                    ):
                        raise ValueError(
                            "execution view must retain a nonempty event list"
                        )
                    events = cast(list[dict[str, Any]], item["events"])
                    if (
                        len(events) != observed.event_count
                        or content_digest(events) != observed.prefix_digest
                        or events[-1]["phase"] != observed.phase
                    ):
                        raise ValueError("execution view prefix digest mismatch")
        elif self.artifact.digest != self.input_digest:  # type: ignore[union-attr]
            raise ValueError("artifact/view digest mismatch")
        if any(subject.snapshot_id != self.snapshot_id for subject in self.subjects):
            raise ValueError("view subjects must belong to its snapshot")
        identity = self.model_dump(mode="json", exclude={"view_id", "artifact"})
        # Include an artifact's transport identity when the input is stored externally.
        if self.artifact is not None:
            identity["artifact"] = self.artifact.model_dump(mode="json")
        if content_digest(identity) != self.view_id:
            raise ValueError("view identity mismatch")
        proof.remember()
        return self


class SignalDefinition(EvidenceRecord):
    signal_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    semantics: Literal[
        "outcome",
        "state_quality",
        "progress",
        "cost",
        "probability",
        "preference",
        "other",
    ]
    description: str = Field(min_length=1)
    units: str = Field(min_length=1)
    direction: Literal["higher", "lower", "neutral"] = "neutral"
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def verify(self) -> Self:
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("signal range is reversed")
        return self


AssessmentStatus = Literal["valid", "inapplicable", "abstained", "failed"]


class AssessmentTarget(EvidenceRecord):
    subject: SubjectRef
    signal: SignalDefinition
    required: bool = True

    @property
    def signal_id(self) -> str:
        return self.signal.signal_id

    @property
    def signal_revision(self) -> str:
        return self.signal.revision

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject.subject_id, self.signal_id, self.signal_revision)


class Derivation(EvidenceRecord):
    rule_id: str = Field(min_length=1)
    rule_revision: str = Field(min_length=1)
    required_parent_ids: tuple[str, ...] = ()
    optional_parent_ids: tuple[str, ...] = ()
    optional_handling: str | None = None
    gate_decision_ids: tuple[str, ...] = ()
    allow_cross_snapshot: bool = False

    @model_validator(mode="after")
    def verify(self) -> Self:
        ids = (*self.required_parent_ids, *self.optional_parent_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate derivation parents")
        if self.optional_parent_ids and not self.optional_handling:
            raise ValueError("optional parents require declared handling")
        return self


class PreferenceResult(EvidenceRecord):
    """A retained comparison, without an implicit numeric reward conversion."""

    alternatives: tuple[SubjectRef, ...]
    relation: Literal["preferred", "equivalent", "incomparable"]
    preferred_subject_id: str | None = None

    @model_validator(mode="after")
    def verify(self) -> Self:
        ids = [subject.subject_id for subject in self.alternatives]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("preference requires at least two distinct alternatives")
        sources = {(s.snapshot_id, s.episode_id) for s in self.alternatives}
        if len(sources) != 1:
            raise ValueError("preference alternatives must share a source")
        if self.relation == "preferred":
            if self.preferred_subject_id not in ids:
                raise ValueError("preferred subject must name a retained alternative")
        elif self.preferred_subject_id is not None:
            raise ValueError("non-preferred relation cannot name a winner")
        return self


class Assessment(EvidenceRecord):
    assessment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    subject: SubjectRef
    view_id: str = Field(min_length=1)
    signal: SignalDefinition
    status: AssessmentStatus
    value: float | None = None
    preference: PreferenceResult | None = None
    invocation_ids: tuple[str, ...] = ()
    reason: str | None = None
    rationale: str | None = None
    evidence: tuple[ArtifactRef, ...] = ()
    derivation: Derivation | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject.subject_id, self.signal.signal_id, self.signal.revision)

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.status == "valid":
            if self.signal.semantics == "preference":
                if self.preference is None or self.value is not None:
                    raise ValueError(
                        "valid preference requires a comparison, not a number"
                    )
                if any(
                    alternative.snapshot_id != self.subject.snapshot_id
                    or alternative.episode_id != self.subject.episode_id
                    for alternative in self.preference.alternatives
                ):
                    raise ValueError("preference alternatives belong to another source")
                if (
                    self.subject.kind != "group"
                    or self.preference.alternatives != self.subject.members
                ):
                    raise ValueError(
                        "preference alternatives must match the requested ordered group"
                    )
                return self
            if self.preference is not None:
                raise ValueError("numeric signal cannot carry a preference")
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(
                    "valid assessments require a finite value, including zero"
                )
            if self.signal.minimum is not None and self.value < self.signal.minimum:
                raise ValueError("value is below signal range")
            if self.signal.maximum is not None and self.value > self.signal.maximum:
                raise ValueError("value is above signal range")
        elif self.value is not None or self.preference is not None:
            raise ValueError(
                "unavailable assessments cannot contain a numeric substitute"
            )
        return self


class ExecutionEvidence(EvidenceRecord):
    """Canonical invocation receipt retained even when result parsing fails."""

    kind: str = Field(min_length=1)
    invocation_id: str | None = Field(default=None, min_length=1)
    payload_json: str
    payload_digest: str

    @classmethod
    def capture(
        cls, kind: str, payload: Any, *, invocation_id: str | None = None
    ) -> Self:
        return cls(
            kind=kind,
            invocation_id=invocation_id,
            payload_json=canonical_json(payload),
            payload_digest=content_digest(payload),
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.payload_json != _canonical(self.payload_json):
            raise ValueError("execution evidence must be canonical JSON")
        if content_digest(json.loads(self.payload_json)) != self.payload_digest:
            raise ValueError("execution evidence digest mismatch")
        return self


class AssessmentRun(EvidenceRecord):
    run_id: str = Field(min_length=1)
    producer_id: str = Field(min_length=1)
    producer_revision: str = Field(min_length=1)
    configuration_json: str = "{}"
    rubric_revision: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    expected: tuple[AssessmentTarget, ...]
    status: Literal[
        "queued", "running", "complete", "partial", "failed", "interrupted"
    ] = "complete"
    reason: str | None = None
    input_evidence: tuple[ArtifactRef, ...] = ()
    output_evidence: tuple[ArtifactRef, ...] = ()
    execution_evidence: tuple[ExecutionEvidence, ...] = ()

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.configuration_json != _canonical(self.configuration_json):
            raise ValueError("configuration must be canonical JSON")
        keys = [target.key for target in self.expected]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate requested target")
        if any(t.subject.snapshot_id != self.snapshot_id for t in self.expected):
            raise ValueError("requested target/source mismatch")
        return self


class AssessmentBatch(EvidenceRecord):
    schema_version: Literal[1] = 1
    source: SourceSnapshot | SourceIdentity
    run: AssessmentRun
    views: tuple[ObservationView, ...]
    assessments: tuple[Assessment, ...] = ()
    dependencies: tuple[Assessment, ...] = ()
    """Immutable accepted upstream records; not new replies to this run's request."""

    # No return annotation: an annotated wrap serializer replaces the field's
    # serialization JSON schema.
    @field_serializer("source", "views", mode="wrap")
    def serialize_archived_evidence(self, value, handler, info):
        return _archive_field(value, handler, info)

    @property
    def missing(self) -> tuple[AssessmentTarget, ...]:
        returned = {assessment.key for assessment in self.assessments}
        return tuple(
            target for target in self.run.expected if target.key not in returned
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.run.snapshot_id != self.source.snapshot_id:
            raise ValueError("run/source mismatch")
        views = {view.view_id: view for view in self.views}
        if len(views) != len(self.views):
            raise ValueError("duplicate view identity")
        records = (*self.dependencies, *self.assessments)
        by_id = {record.assessment_id: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("duplicate assessment identity")
        expected = {target.key: target for target in self.run.expected}
        replies: set[tuple[str, str, str]] = set()
        source_nodes = {
            (node.trace_id, node.node_index): node.node_content_digest
            for node in self.source.nodes
        }
        source_executions = _membership(self.source.executions)

        def check_subject(subject: SubjectRef) -> None:
            if (
                subject.snapshot_id != self.source.snapshot_id
                or subject.episode_id != self.source.episode_id
            ):
                raise ValueError("subject/source mismatch")
            if subject.node_index is not None:
                if subject.trace_id is None:
                    raise ValueError("node subject requires a trace identity")
                key = (subject.trace_id, subject.node_index)
                if source_nodes.get(key) != subject.node_content_digest:
                    raise ValueError("subject node is absent or rewritten")
            if (
                subject.trace_id is not None
                and subject.trace_id not in self.source.trace_ids
            ):
                raise ValueError("subject belongs to an undeclared trace")
            if subject.execution is not None and not source_executions(
                subject.execution
            ):
                raise ValueError("execution prefix is absent or rewritten")
            for member in subject.members:
                check_subject(member)

        for view in self.views:
            if view.snapshot_id != self.source.snapshot_id:
                raise ValueError("view/source mismatch")
            for subject in view.subjects:
                check_subject(subject)
            if view.builder_revision == "native_execution_prefix_v1" and isinstance(
                self.source, SourceSnapshot
            ):
                from verifiers.v1.assessment_source import resolve_execution

                if view.input_json is None:
                    raise ValueError("execution view must retain input JSON")
                for item in json.loads(view.input_json):
                    observed = ExecutionRef.model_validate(item["observed"])
                    if list(resolve_execution(self.source, observed)) != item["events"]:
                        raise ValueError(
                            "execution view changed retained source prefix"
                        )
        for target in self.run.expected:
            check_subject(target.subject)
        for assessment in self.assessments:
            if assessment.run_id != self.run.run_id:
                raise ValueError("reply belongs to another run")
            check_subject(assessment.subject)
            if assessment.preference is not None:
                for alternative in assessment.preference.alternatives:
                    check_subject(alternative)
            if len(assessment.invocation_ids) != len(set(assessment.invocation_ids)):
                raise ValueError("duplicate result invocation link")
            recorded_invocations = {
                receipt.invocation_id
                for receipt in self.run.execution_evidence
                if receipt.invocation_id is not None
            }
            if set(assessment.invocation_ids) - recorded_invocations:
                raise ValueError("result invocation has no retained execution evidence")
            view = views.get(assessment.view_id)
            if view is None or assessment.subject not in view.subjects:
                raise ValueError("reply must reference its retained observation view")
            if assessment.key not in expected or assessment.key in replies:
                raise ValueError("unexpected or duplicate reply target")
            if assessment.signal != expected[assessment.key].signal:
                raise ValueError("reply changed the requested signal definition")
            replies.add(assessment.key)
        if self.run.status == "complete" and self.missing:
            raise ValueError("complete run must account for every requested target")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(record: Assessment) -> None:
            if record.assessment_id in visiting:
                raise ValueError("cyclic assessment derivation")
            if record.assessment_id in visited:
                return
            visiting.add(record.assessment_id)
            if derivation := record.derivation:
                for parent_id in (
                    *derivation.required_parent_ids,
                    *derivation.optional_parent_ids,
                    *derivation.gate_decision_ids,
                ):
                    parent = by_id.get(parent_id)
                    if parent is None:
                        raise ValueError("unresolved assessment dependency")
                    if (
                        parent.subject.snapshot_id != record.subject.snapshot_id
                        or parent.subject.episode_id != record.subject.episode_id
                    ) and not derivation.allow_cross_snapshot:
                        raise ValueError("undeclared cross-snapshot dependency")
                    if (
                        parent_id in derivation.required_parent_ids
                        and parent.status != "valid"
                        and record.status == "valid"
                    ):
                        raise ValueError(
                            "valid derivation has unavailable required input"
                        )
                    if (
                        parent_id in derivation.gate_decision_ids
                        and parent.status != "valid"
                    ):
                        raise ValueError("gate decision must be valid")
                    visit(parent)
            visiting.remove(record.assessment_id)
            visited.add(record.assessment_id)

        for record in records:
            visit(record)
        return self


class AssessmentRequest(EvidenceRecord):
    source: SourceIdentity
    run: AssessmentRun
    views: tuple[ObservationView, ...]

    @model_validator(mode="after")
    def verify(self) -> Self:
        if not self.views:
            raise ValueError("assessment requires declared raw context")
        if any(v.scope == "prefix" for v in self.views) and any(
            v.scope != "prefix" for v in self.views
        ):
            raise ValueError(
                "prefix context cannot also expose later observation views"
            )
        # Reuse the same shape checks as a partial response without inventing zeros.
        if any(v.scope == "prefix" for v in self.views) and len(self.views) != 1:
            raise ValueError(
                "prefix context needs one explicit shared visibility boundary"
            )
        AssessmentBatch(
            source=self.source,
            run=self.run.model_copy(update={"status": "partial"}),
            views=self.views,
        )
        return self


class AssessmentContext(EvidenceRecord):
    """Available declared raw material; callers may freely transform working copies."""

    views: tuple[ObservationView, ...]
    dependencies: tuple[Assessment, ...] = ()
    _execution_evidence: list[ExecutionEvidence] = PrivateAttr(default_factory=list)
    _sealed_source: SourceSnapshot | None = PrivateAttr(default=None)

    def retrospective_source(self) -> SourceSnapshot:
        """Read the executor's sealed source only for retrospective assessment.

        Prefix and action-result views must not gain future context through this
        accessor. Caller-authored input views cannot replace the runtime anchor.
        """
        if not self.views or any(view.scope != "retrospective" for view in self.views):
            raise ValueError("sealed source access requires retrospective views")
        if self._sealed_source is None:
            raise ValueError("sealed source is unavailable outside native execution")
        return SourceSnapshot.model_validate(
            self._sealed_source.model_dump(mode="python")
        )

    def record_evidence(
        self, kind: str, payload: Any, *, invocation_id: str | None = None
    ) -> ExecutionEvidence:
        """Record request/response/usage before parsing; no scalar result is implied.

        This is an invocation-local journal. The executor attaches immutable
        receipts to native batches on success, failure, or interruption. Payloads
        must be explicitly selected; never include transport credentials.
        """
        receipt = ExecutionEvidence.capture(kind, payload, invocation_id=invocation_id)
        self._execution_evidence.append(receipt)
        return receipt

    @property
    def execution_evidence(self) -> tuple[ExecutionEvidence, ...]:
        return tuple(self._execution_evidence)

    def input(self, view_id: str) -> Any:
        view = next((item for item in self.views if item.view_id == view_id), None)
        if view is None:
            raise KeyError(view_id)
        if view.input_json is None:
            raise ValueError(
                "external artifact must be resolved by its authorized transport"
            )
        # Each call yields a fresh decoded copy, never mutable internal storage.
        return json.loads(view.input_json)


class Assessor(Protocol):
    async def assess(
        self, request: AssessmentRequest, context: AssessmentContext
    ) -> AssessmentBatch | Iterable[Assessment] | AsyncIterable[Assessment]: ...
