import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)
from renderers.base import MultiModalData
from typing_extensions import TypedDict


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(BaseModel):
    url: str


class ImageUrlContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


ContentPart = Annotated[
    TextContentPart | ImageUrlContentPart, Field(discriminator="type")
]
MessageContent = str | list[ContentPart]
"""Plain text or typed multimodal content parts."""


def content_to_parts(content) -> MessageContent:
    """Type OpenAI content parts, dropping unsupported part types."""
    if not isinstance(content, list):
        return content or ""
    parts: list[ContentPart] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append(TextContentPart(text=p.get("text", "")))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            parts.append(ImageUrlContentPart(image_url=ImageUrlSource(url=url)))
    return parts


def content_text(content: "MessageContent | None") -> str:
    """Extract text from message content, dropping images."""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.text for part in content or [] if isinstance(part, TextContentPart)
    )


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: MessageContent


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(BaseModel):
    id: str
    type: Literal["function", "custom"] = "function"
    name: str
    arguments: str
    """Provider arguments or parser output passed to dispatch. GeneratedCallAttempt.raw
    retains original generated syntax when a renderer normalizes arguments."""


def generated_completion_digest(token_ids: Iterable[int]) -> str:
    """Hash original completion tokens, without rendering or normalizing their text."""
    retained = list(token_ids)
    if any(type(token) is not int for token in retained):
        raise ValueError("generated completion identity requires integer token IDs")
    encoded = json.dumps(
        retained, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class GeneratedCallProducer(BaseModel):
    """Inspectable parser provenance; a descriptor does not establish span exactness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parser_revision: str = Field(min_length=1)
    descriptor_json: str
    descriptor_digest: str = Field(min_length=1)

    @classmethod
    def capture(cls, parser_revision: str, descriptor: dict[str, Any]) -> Self:
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return cls(
            parser_revision=parser_revision,
            descriptor_json=encoded,
            descriptor_digest=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )

    @model_validator(mode="after")
    def verify(self) -> Self:
        decoded = json.loads(self.descriptor_json)
        if not isinstance(decoded, dict):
            raise ValueError("generated-call producer descriptor must be a JSON object")  # noqa: TRY004
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical != self.descriptor_json:
            raise ValueError(
                "generated-call producer descriptor must be canonical JSON"
            )
        if (
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            != self.descriptor_digest
        ):
            raise ValueError("generated-call producer descriptor digest mismatch")
        return self


class GeneratedCallAttempt(BaseModel):
    """One parser-observed generated action, independent of dispatch or execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_index: int = Field(ge=0, strict=True)
    emitted_call_index: int | None = Field(default=None, ge=0, strict=True)
    provider_call_id: str | None = None
    parse_status: str = Field(min_length=1)
    raw: str
    name: str | None = None
    arguments: str | None = None
    token_span: tuple[StrictInt, StrictInt] | None = None
    coordinate_system: Literal["completion", "node_local_full_tokens"]
    parser_revision: str = Field(min_length=1)
    span_fidelity: Literal["exact", "joint", "unavailable"]
    completion_token_digest: str = Field(min_length=1)

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.token_span is not None:
            start, end = self.token_span
            if start < 0 or end <= start:
                raise ValueError(
                    "generated-call span must be a nonempty half-open interval"
                )
        if self.span_fidelity == "unavailable":
            if self.token_span is not None:
                raise ValueError(
                    "unavailable generated-call support cannot carry a span"
                )
        elif self.token_span is None:
            raise ValueError(
                "exact/joint generated-call support requires a retained span"
            )
        return self


def generated_arguments_equal(first: str | None, second: str | None) -> bool:
    """Compare JSON argument meaning without replacing retained original strings.

    JSON encoding preserves booleans versus numbers; Python value equality does not.
    Non-JSON custom or malformed input is comparable only by its exact raw string.
    """
    first = first if first is not None else "{}"
    second = second if second is not None else "{}"
    try:
        first_json = json.dumps(
            json.loads(first), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        second_json = json.dumps(
            json.loads(second), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (ValueError, TypeError):
        return first == second
    return first_json == second_json


def validate_generated_call_links(
    attempts: tuple[GeneratedCallAttempt, ...],
    tool_calls: list[ToolCall] | tuple[ToolCall, ...] | None,
) -> None:
    """Verify parser-attempt correlation with emitted messages, not execution."""
    emitted = tool_calls or ()
    used: set[int] = set()
    for item in attempts:
        attempt = GeneratedCallAttempt.model_validate(item.model_dump(mode="json"))
        index = attempt.emitted_call_index
        if index is None:
            continue
        if index in used or index >= len(emitted):
            raise ValueError(
                "generated-call emitted index is duplicate or out of bounds"
            )
        used.add(index)
        call = emitted[index]
        if call.type != "function":
            raise ValueError(
                "generated function attempt cannot link to a custom emitted call"
            )
        if attempt.name != call.name:
            raise ValueError("generated-call name does not match emitted call")
        if not generated_arguments_equal(attempt.arguments, call.arguments):
            raise ValueError("generated-call arguments do not match emitted call")
        if attempt.provider_call_id is not None and attempt.provider_call_id != call.id:
            raise ValueError(
                "generated-call provider identity does not match emitted call"
            )


def validate_generated_call_spans(attempts: tuple[GeneratedCallAttempt, ...]) -> None:
    """Exact support is isolated; jointly attributed parser spans may overlap."""
    validated = tuple(
        GeneratedCallAttempt.model_validate(item.model_dump(mode="json"))
        for item in attempts
    )
    for index, left in enumerate(validated):
        if left.token_span is None:
            continue
        for right in validated[index + 1 :]:
            if right.token_span is None:
                continue
            if left.coordinate_system != right.coordinate_system:
                raise ValueError("generated-call span coordinates cannot be mixed")
            overlap = max(left.token_span[0], right.token_span[0]) < min(
                left.token_span[1], right.token_span[1]
            )
            if overlap and "exact" in {left.span_fidelity, right.span_fidelity}:
                raise ValueError(
                    "exact generated-call span overlaps another generated attempt"
                )


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    provider_state: list[dict[str, Any]] | None = None
    """Opaque native items replayed to preserve signed or encrypted reasoning state."""


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: MessageContent
    name: str | None = None
    """Needed by templates such as Harmony when bridge tails omit the issuing call."""


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
Messages = list[Message]


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


class Request(BaseModel):
    """The typed conversation about to cross a model or harness boundary."""

    messages: Messages
    tools: list[Tool] | None = None


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(BaseModel):
    """Provider token accounting.

    `prompt_tokens` excludes cache reads; `input_tokens` adds them back. Reasoning tokens
    are a subset of completion tokens and are not added to totals again.
    """

    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None

    @classmethod
    def from_openai(cls, usage: Any | None) -> "Usage | None":
        """Build usage while splitting cached tokens out of `prompt_tokens`."""
        if usage is None:
            return None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached = prompt_details.cached_tokens if prompt_details else None
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning = completion_details.reasoning_tokens if completion_details else None
        return cls(
            prompt_tokens=usage.prompt_tokens - (cached or 0),
            completion_tokens=usage.completion_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
            cost=getattr(usage, "cost", None),
        )

    @classmethod
    def aggregate(cls, usages: Iterable["Usage"]) -> "Usage | None":
        """Sum per-response usage while preserving whether cache usage was reported."""
        values = list(usages)
        if not values:
            return None
        # For the optional fields (cached / reasoning / cost), sum the responses that report them
        # and yield None only when *no* response does — so one response omitting a field (e.g. a
        # judge whose provider doesn't report reasoning or cost) doesn't null out the whole total.
        cached = [
            u.cached_input_tokens for u in values if u.cached_input_tokens is not None
        ]
        reasoning = [
            u.reasoning_tokens for u in values if u.reasoning_tokens is not None
        ]
        costs = [u.cost for u in values if u.cost is not None]
        return cls(
            prompt_tokens=sum(usage.prompt_tokens for usage in values),
            completion_tokens=sum(usage.completion_tokens for usage in values),
            cached_input_tokens=sum(cached) if cached else None,
            reasoning_tokens=sum(reasoning) if reasoning else None,
            cost=sum(costs) if costs else None,
        )

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens + (self.cached_input_tokens or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.completion_tokens


class RoutedExperts(TypedDict):
    """Base64 uint8 `[tokens, layers, top_k]` routing and its prompt offset."""

    data: Any
    shape: list[int]
    start: int


@dataclass
class SamplingMask:
    """Sampling masks stored as flat int32 `ids` and `counts` arrays.

    Each row contains the token ids that survived sampling filters for one completion
    token. Row boundaries are recovered from `counts`.
    """

    ids: Any
    counts: Any

    @classmethod
    def from_sampling_mask(cls, sampling_mask: list[list[int]]) -> "SamplingMask":
        counts = np.fromiter(
            (len(row) for row in sampling_mask),
            dtype=np.int32,
            count=len(sampling_mask),
        )
        ids = (
            np.concatenate([np.asarray(row, dtype=np.int32) for row in sampling_mask])
            if int(counts.sum())
            else np.empty(0, dtype=np.int32)
        )
        return cls(ids=ids, counts=counts)


class TurnTokens(BaseModel):
    """Training tokens from renderer tokenization or provider-returned token IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)
    # Original parser attempts are transferred to the committed assistant node.
    generated_calls: tuple[GeneratedCallAttempt, ...] = Field(default=(), exclude=True)
    generated_call_producer: GeneratedCallProducer | None = Field(
        default=None, exclude=True
    )

    # Transient carrier (excluded): per-message token spans into `prompt_ids` from the renderer,
    # consumed by the turn's `commit` to attribute tokens per message, then dropped.
    message_spans: list[tuple[int, int] | None] | None = Field(
        default=None, exclude=True
    )
    is_content: list[bool] | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the renderer's multimodal sidecar (image tensors + offsets),
    # attributed per node by the turn's `commit`, then dropped — never persisted.
    multi_modal_data: MultiModalData | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the renderer's special-token id -> modality marker map,
    # stamped onto `Trace.mm_token_type_id_map` by the turn's `commit`. None unless the
    # rendering renderer is multimodal.
    mm_token_type_id_map: dict[int, int] | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): the MoE expert-routing data from `generate` (expert ids
    # per token), attributed per node by the turn's `commit` into `MessageNode.routed_experts`,
    # then dropped. None unless the engine ran with `enable_return_routed_experts`.
    routed_experts: RoutedExperts | None = Field(default=None, exclude=True)
    # Transient carrier (excluded): per-completion-token sampling masks,
    # attributed to the assistant node by the turn's `commit`, then dropped.
    sampling_mask: SamplingMask | None = Field(default=None, exclude=True)


class Response(BaseModel):
    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    raw: dict | None = Field(default=None, exclude=True, repr=False)
    """Full native response object returned to the program; excluded from traces."""


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )

    def wire_args(self) -> dict[str, Any]:
        """Flatten OpenAI-style ``extra_body`` before building a provider request."""
        args = self.model_dump(exclude_none=True)
        extra_body = {
            key: value
            for key, value in (args.pop("extra_body", None) or {}).items()
            if value is not None
        }
        if "max_tokens" in args:
            extra_body.pop("max_completion_tokens", None)
        return {**extra_body, **args}


Sampling = SamplingConfig


ID = str
"""Plugin id: the name of an installed package exporting the plugin."""
