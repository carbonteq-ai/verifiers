"""Renderer parsers needed by supported training-client model protocols."""

from __future__ import annotations

import ast
from typing import Any

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


def register_renderer_extensions() -> None:
    """Register CarbonTeq-supported protocol parsers idempotently."""

    from renderers.parsers import TOOL_PARSERS

    TOOL_PARSERS.setdefault("lfm2", LFM2ToolParser)


__all__ = ["LFM2ToolParser", "register_renderer_extensions"]
