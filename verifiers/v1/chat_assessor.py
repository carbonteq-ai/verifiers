"""Model assessment over declared views through the existing Judge transport."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from verifiers.v1.assessments import Assessment, AssessmentContext, AssessmentRequest
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.types import Messages


class ChatAssessor:
    """Construct a view-only prompt, retain the exchange, then parse locally.

    Parsing produces declared Assessment records; the common executor validates
    identity, signal meaning, coverage and values. No score reduction or retry is
    performed here. SDK structured parsing is deliberately not used: raw output
    must be retained before malformed verdicts can fail validation.
    """

    def __init__(
        self,
        judge: Judge,
        build_messages: Callable[
            [AssessmentRequest, AssessmentContext],
            str | Messages | Awaitable[str | Messages],
        ],
        parse: Callable[[JudgeResponse[Any], AssessmentRequest], Iterable[Assessment]],
    ) -> None:
        self.judge = judge
        self.build_messages = build_messages
        self.parse = parse

    async def assess(
        self, request: AssessmentRequest, context: AssessmentContext
    ) -> tuple[Assessment, ...]:
        messages = self.build_messages(request, context)
        if inspect.isawaitable(messages):
            messages = await messages
        invocation_id = uuid.uuid4().hex
        wire = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else [message_to_wire(message) for message in messages]
        )
        context.record_evidence(
            "chat_request",
            {
                "model": self.judge.config.model,
                "messages": wire,
                "sampling": self.judge.config.sampling.model_dump(
                    mode="json", exclude_none=True
                ),
            },
            invocation_id=invocation_id,
        )
        try:
            response = await self.judge.complete(
                messages, trace=None, schema=None, parse=None
            )
        except asyncio.CancelledError:
            context.record_evidence(
                "chat_interrupted",
                {"status": "cancelled", "usage_known": False},
                invocation_id=invocation_id,
            )
            raise
        except Exception as error:
            context.record_evidence(
                "chat_failed",
                {"error_type": type(error).__name__, "usage_known": False},
                invocation_id=invocation_id,
            )
            raise
        context.record_evidence(
            "chat_response",
            response.model_dump(mode="json"),
            invocation_id=invocation_id,
        )
        return tuple(
            result.model_copy(
                update={
                    "invocation_ids": tuple(
                        dict.fromkeys((*result.invocation_ids, invocation_id))
                    )
                }
            )
            for result in self.parse(response, request)
        )
