"""Opt-in tool-server observations, independent of world-state replacement."""

from __future__ import annotations

import contextvars
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from verifiers.v1.assessments import canonical_json

STATE_REVISION_HEADER = "X-Verifiers-State-Revision"
STATE_EXPECTED_REVISION_HEADER = "X-Verifiers-State-Expected-Revision"
STATE_CONFLICT_HEADER = "X-Verifiers-State-Conflict"
STATE_WRITE_ID_HEADER = "X-Verifiers-State-Write-ID"
MAX_EXECUTION_RECEIPT_BYTES = 16 * 1024 * 1024
ExecutionPhase = Literal["dispatch", "returned", "raised", "interrupted"]
StatePersistence = Literal["not_attempted", "unchanged", "applied", "failed", "unknown"]
EXECUTION_METADATA_KEY = "verifiers.execution"


class DispatchMetadata(BaseModel):
    """Host-issued dispatch capability carried outside model-visible arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    dispatch_ticket: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    transport_attempt_index: int = Field(ge=0)


class ToolServerReceipt(BaseModel):
    """Reported tool lifecycle; returned does not imply persistence or domain success."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invocation_id: str = Field(min_length=1)
    event_index: int = Field(ge=0, le=1, strict=True)
    phase: ExecutionPhase
    tool_name: str = Field(min_length=1)
    arguments_json: str
    evidence_json: tuple[str, ...] = ()
    result_json: str | None = None
    error_json: str | None = None
    state_read_revision: int | None = Field(default=None, ge=0, strict=True)
    state_write_revision: int | None = Field(default=None, ge=0, strict=True)
    state_conflict: bool | None = None
    state_persistence: StatePersistence = "not_attempted"
    state_error_json: str | None = None
    parent_execution_id: str | None = Field(default=None, min_length=1, strict=True)
    dispatch_ticket: str | None = Field(default=None, min_length=1, strict=True)
    transport_attempt_index: int | None = Field(default=None, ge=0, strict=True)
    server_name: str | None = Field(default=None, strict=True)

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        data = handler(self)
        for key in (
            "parent_execution_id",
            "dispatch_ticket",
            "transport_attempt_index",
            "server_name",
        ):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def verify(self):
        link = (
            self.parent_execution_id,
            self.dispatch_ticket,
            self.transport_attempt_index,
            self.server_name,
        )
        if any(value is not None for value in link) and any(
            value is None for value in link
        ):
            raise ValueError("linked receipt requires complete dispatch provenance")
        for encoded in (
            self.arguments_json,
            *self.evidence_json,
            self.result_json,
            self.error_json,
            self.state_error_json,
        ):
            if encoded is not None and canonical_json(json.loads(encoded)) != encoded:
                raise ValueError("tool-server receipt material must be canonical JSON")
        if self.phase == "dispatch":
            if (
                self.event_index != 0
                or self.result_json is not None
                or self.error_json is not None
                or self.state_persistence != "not_attempted"
                or self.state_write_revision is not None
                or self.state_conflict is not None
                or self.state_error_json is not None
            ):
                raise ValueError("dispatch receipt cannot claim a terminal outcome")
        elif self.event_index != 1:
            raise ValueError("terminal tool-server receipt requires event index one")
        if self.phase == "returned":
            if self.result_json is None or self.error_json is not None:
                raise ValueError("returned receipt requires its original result")
        elif self.result_json is not None:
            raise ValueError("non-returned receipt cannot carry a returned result")
        if self.phase in ("raised", "interrupted") and self.error_json is None:
            raise ValueError(
                "raised/interrupted receipt requires observed error material"
            )
        if self.phase in ("raised", "interrupted") and (
            self.state_persistence != "not_attempted"
            or self.state_write_revision is not None
            or self.state_conflict is not None
            or self.state_error_json is not None
        ):
            raise ValueError(
                "raised/interrupted tool cannot claim a persisted world write"
            )
        if self.state_persistence == "applied" and self.state_write_revision is None:
            raise ValueError(
                "applied state requires an acknowledged revision; use unknown for legacy metadata"
            )
        if self.state_persistence != "applied" and (
            self.state_write_revision is not None or self.state_conflict is not None
        ):
            raise ValueError(
                "write revision/conflict require acknowledged applied persistence"
            )
        if (
            self.state_persistence in ("unchanged", "applied", "not_attempted")
            and self.state_error_json is not None
        ):
            raise ValueError(
                "successful/unattempted persistence cannot carry a state error"
            )
        return self


@dataclass
class StateRevision:
    read: int | None = None
    written: int | None = None
    conflict: bool | None = None
    persistence: StatePersistence = "not_attempted"


@dataclass
class ExecutionCapture:
    tool_name: str
    arguments_json: str
    revision: StateRevision
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    evidence: list[str] = field(default_factory=list)
    parent_execution_id: str | None = None
    dispatch_ticket: str | None = None
    transport_attempt_index: int | None = None
    server_name: str | None = None

    def receipt(
        self,
        phase: ExecutionPhase,
        *,
        result: Any = None,
        error: BaseException | None = None,
        state_error: BaseException | None = None,
    ) -> ToolServerReceipt:
        def error_json(value: BaseException | None) -> str | None:
            return (
                canonical_json({"type": type(value).__name__, "message": str(value)})
                if value is not None
                else None
            )

        return ToolServerReceipt(
            invocation_id=self.invocation_id,
            event_index=0 if phase == "dispatch" else 1,
            phase=phase,
            tool_name=self.tool_name,
            arguments_json=self.arguments_json,
            evidence_json=tuple(self.evidence),
            result_json=canonical_json(result) if phase == "returned" else None,
            error_json=error_json(error),
            state_read_revision=self.revision.read,
            state_write_revision=self.revision.written,
            state_conflict=self.revision.conflict,
            state_persistence=self.revision.persistence,
            state_error_json=error_json(state_error),
            parent_execution_id=self.parent_execution_id,
            dispatch_ticket=self.dispatch_ticket,
            transport_attempt_index=self.transport_attempt_index,
            server_name=self.server_name,
        )


active_execution: contextvars.ContextVar[ExecutionCapture | None] = (
    contextvars.ContextVar("vf_execution_capture", default=None)
)
active_revision: contextvars.ContextVar[StateRevision | None] = contextvars.ContextVar(
    "vf_state_revision", default=None
)


def record_execution_evidence(value: Any) -> bool:
    """Append neutral JSON raw material to this invocation; inactive capture returns False."""
    capture = active_execution.get()
    if capture is None:
        return False
    capture.evidence.append(canonical_json(value))
    return True
