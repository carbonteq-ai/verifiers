"""Renderer parsers needed by supported training-client model protocols."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from renderers import RenderedTokens
from renderers.base import ParsedToolCall, ToolCallParseStatus


class LFM2ToolParser:
    """Parse LFM2's ``[python_call(...)]`` tool-call token block.

    LFM emits a Python-expression surface syntax, but parsing is deliberately
    data-only: only a list of named calls, keyword arguments, and
    ``ast.literal_eval`` values are accepted.  No sampled code is executed.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._start = self._token_id("<|tool_call_start|>")
        self._end = self._token_id("<|tool_call_end|>")

    def _token_id(self, token: str) -> int | None:
        value = self._tokenizer.convert_tokens_to_ids(token)
        unknown = getattr(self._tokenizer, "unk_token_id", None)
        return value if isinstance(value, int) and value != unknown else None

    def extract(self, token_ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        if self._start is None or self._start not in token_ids:
            return token_ids, []
        start = token_ids.index(self._start)
        if self._end is None or self._end not in token_ids[start + 1 :]:
            raw = self._tokenizer.decode(token_ids[start + 1 :], skip_special_tokens=False)
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=(start, len(token_ids)),
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            ]
        end = token_ids.index(self._end, start + 1)
        raw = self._tokenizer.decode(token_ids[start + 1 : end], skip_special_tokens=False).strip()
        span = (start, end + 1)
        try:
            expression = ast.parse(raw, mode="eval").body
        except SyntaxError:
            return token_ids[:start], [
                ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.MALFORMED_STRUCTURE)
            ]
        if not isinstance(expression, ast.List) or not expression.elts:
            return token_ids[:start], [
                ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.MALFORMED_STRUCTURE)
            ]
        calls: list[ParsedToolCall] = []
        for item in expression.elts:
            if (
                not isinstance(item, ast.Call)
                or not isinstance(item.func, ast.Name)
                or item.args
                or any(keyword.arg is None for keyword in item.keywords)
            ):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(raw, item) or raw,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            keys = [str(keyword.arg) for keyword in item.keywords]
            if len(keys) != len(set(keys)):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(raw, item) or raw,
                        name=item.func.id,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            try:
                arguments = {
                    str(keyword.arg): ast.literal_eval(keyword.value)
                    for keyword in item.keywords
                }
            except (TypeError, ValueError):
                calls.append(
                    ParsedToolCall(
                        raw=ast.get_source_segment(raw, item) or raw,
                        name=item.func.id,
                        token_span=span,
                        status=ToolCallParseStatus.INVALID_JSON,
                    )
                )
                continue
            calls.append(
                ParsedToolCall(
                    raw=ast.get_source_segment(raw, item) or raw,
                    name=item.func.id,
                    arguments=arguments,
                    token_span=span,
                )
            )
        return [*token_ids[:start], *token_ids[end + 1 :]], calls


class K2IFMToolParser:
    """Parse K2-Horizon's IFM XML tool-call block from sampled tokens."""

    _call = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", flags=re.DOTALL)
    _argument = re.compile(
        r"\s*<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        flags=re.DOTALL,
    )

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._start = self._marker("<ifm|tool_calls>")
        self._end = self._marker("</ifm|tool_calls>")

    def _marker(self, text: str) -> list[int]:
        return [int(token) for token in self._tokenizer.encode(text, add_special_tokens=False)]

    @staticmethod
    def _find(tokens: list[int], marker: list[int], start: int = 0) -> int | None:
        if not marker:
            return None
        limit = len(tokens) - len(marker) + 1
        return next((index for index in range(start, limit) if tokens[index : index + len(marker)] == marker), None)

    def extract(self, token_ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        start = self._find(token_ids, self._start)
        if start is None:
            return token_ids, []
        body_start = start + len(self._start)
        end = self._find(token_ids, self._end, body_start)
        if end is None:
            raw = self._tokenizer.decode(token_ids[body_start:], skip_special_tokens=False)
            return token_ids[:start], [
                ParsedToolCall(
                    raw=raw,
                    token_span=(start, len(token_ids)),
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            ]
        block_end = end + len(self._end)
        raw_block = self._tokenizer.decode(token_ids[body_start:end], skip_special_tokens=False)
        span = (start, block_end)
        calls: list[ParsedToolCall] = []
        matches = list(self._call.finditer(raw_block))
        if not matches:
            calls.append(
                ParsedToolCall(raw=raw_block, token_span=span, status=ToolCallParseStatus.MALFORMED_STRUCTURE)
            )
        for match in matches:
            raw = match.group(1).strip()
            name, separator, remainder = raw.partition("\n")
            name = name.strip()
            if not separator or not name:
                calls.append(
                    ParsedToolCall(raw=raw, token_span=span, status=ToolCallParseStatus.MISSING_NAME)
                )
                continue
            arguments: dict[str, Any] = {}
            position = 0
            valid = True
            for argument in self._argument.finditer(remainder):
                if remainder[position : argument.start()].strip():
                    valid = False
                    break
                key = argument.group(1).strip()
                value_text = argument.group(2).strip()
                if not key or key in arguments:
                    valid = False
                    break
                try:
                    arguments[key] = json.loads(value_text)
                except json.JSONDecodeError:
                    arguments[key] = value_text
                position = argument.end()
            if remainder[position:].strip() or not valid:
                calls.append(
                    ParsedToolCall(
                        raw=raw,
                        name=name,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                continue
            calls.append(ParsedToolCall(raw=raw, name=name, arguments=arguments, token_span=span))
        return [*token_ids[:start], *token_ids[block_end:]], calls


class K2IFMReasoningParser:
    """Preserve K2's generated thinking field across tool-result turns."""

    _close = re.compile(r"</ifm\|(think|think_fast|think_faster)>")
    _open = re.compile(r"^\s*<ifm\|(think|think_fast|think_faster)>\s*")

    def __init__(self, tokenizer: Any) -> None:
        del tokenizer

    def extract(self, text: str) -> tuple[str | None, str]:
        close = self._close.search(text)
        if close is None:
            return None, text
        before = text[: close.start()]
        opened = self._open.match(before)
        reasoning = before[opened.end() :] if opened is not None else before
        return reasoning or None, text[close.end() :]


def bridge_lfm2_tool_cycle(
    renderer: Any,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[dict[str, Any]],
) -> RenderedTokens | None:
    """Append LFM tool observations without retokenizing sampled history."""

    if not new_messages or any(message.get("role") != "tool" for message in new_messages):
        return None
    tokenizer = getattr(renderer, "_tokenizer", None)
    if tokenizer is None or not previous_completion_ids:
        return None
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(bos_token_id, int) or not isinstance(eos_token_id, int):
        return None
    suffix = renderer.render(new_messages, tools=None, add_generation_prompt=True)
    if not suffix.token_ids or int(suffix.token_ids[0]) != bos_token_id:
        return None
    close_ids = [] if previous_completion_ids[-1] == eos_token_id else [eos_token_id]
    newline_ids = [int(value) for value in tokenizer.encode("\n", add_special_tokens=False)]
    if not newline_ids:
        return None
    prefix_length = len(previous_prompt_ids) + len(previous_completion_ids) + len(close_ids) + len(newline_ids)
    suffix_is_content = list(suffix.is_content[1:]) if suffix.is_content else []
    suffix_sampled = list(suffix.sampled_mask[1:]) if suffix.sampled_mask else []
    return RenderedTokens(
        token_ids=[
            *previous_prompt_ids,
            *previous_completion_ids,
            *close_ids,
            *newline_ids,
            *suffix.token_ids[1:],
        ],
        message_indices=[-1] * prefix_length + list(suffix.message_indices[1:]),
        sampled_mask=([False] * prefix_length + suffix_sampled) if suffix_sampled else [],
        is_content=([False] * prefix_length + suffix_is_content) if suffix_is_content else [],
        message_roles=list(suffix.message_roles),
        message_tool_names=list(suffix.message_tool_names),
    )


def register_renderer_extensions() -> None:
    """Register CarbonTeq-supported protocol parsers idempotently."""

    from renderers.parsers import REASONING_PARSERS, TOOL_PARSERS

    TOOL_PARSERS.setdefault("lfm2", LFM2ToolParser)
    TOOL_PARSERS.setdefault("k2-ifm", K2IFMToolParser)
    REASONING_PARSERS.setdefault("k2-ifm", K2IFMReasoningParser)


__all__ = [
    "K2IFMReasoningParser",
    "K2IFMToolParser",
    "LFM2ToolParser",
    "bridge_lfm2_tool_cycle",
    "register_renderer_extensions",
]
