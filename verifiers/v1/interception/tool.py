"""Typed wire payload for native harness tool hooks."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_serializer,
    model_validator,
)

from verifiers.v1.assessments import canonical_json
from verifiers.v1.types import ToolCall, ToolMessage

ToolExecutionPhase = Literal[
    "before", "after", "dispatch", "raised", "rejected", "interrupted"
]


class MCPDispatch(BaseModel):
    """Host routing of one original call, outside model-visible arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    server_name: StrictStr
    tool_name: StrictStr = Field(min_length=1)
    arguments_json: StrictStr

    @model_validator(mode="after")
    def verify(self):
        import json

        value = json.loads(self.arguments_json)
        if not isinstance(value, dict) or canonical_json(value) != self.arguments_json:
            raise ValueError("MCP dispatch arguments require canonical JSON object")
        return self


class ToolHookRequest(BaseModel):
    phase: ToolExecutionPhase
    message: ToolMessage
    execution_id: str | None = Field(default=None, min_length=1)
    call: ToolCall | None = None
    event_index: int | None = Field(default=None, ge=0, strict=True)
    error: str | dict | None = None
    raw_result: str | None = None
    mcp_dispatch: MCPDispatch | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        value = handler(self)
        if value.get("mcp_dispatch") is None:
            value.pop("mcp_dispatch", None)
        return value

    @model_validator(mode="after")
    def verify_occurrence(self):
        if self.mcp_dispatch is not None and self.phase != "dispatch":
            raise ValueError("MCP destination is only meaningful at dispatch")
        explicit = (
            self.execution_id is not None,
            self.call is not None,
            self.event_index is not None,
        )
        if any(explicit) and not all(explicit):
            raise ValueError(
                "tool execution identity, call and event index must be supplied together"
            )
        if self.phase not in ("before", "after") and not all(explicit):
            raise ValueError(
                "tool lifecycle phases require an explicit execution occurrence"
            )
        if self.call is not None and self.call.id != self.message.tool_call_id:
            raise ValueError("tool execution call and message IDs disagree")
        if self.error is not None and self.phase not in ("raised", "interrupted"):
            raise ValueError(
                "tool execution error is only meaningful for raised/interrupted phases"
            )
        if self.raw_result is not None and self.phase not in ("after", "rejected"):
            raise ValueError(
                "raw harness-result text is only meaningful for after/rejected phases"
            )
        canonical_json(self.model_dump(mode="json"))
        return self
