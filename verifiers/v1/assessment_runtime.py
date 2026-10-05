"""Execute assessors against retained views and keep every attempt's evidence."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, Callable, Iterable, Mapping
from typing import Any

from verifiers.v1._validation_scope import planned_validation_child, validation_owner
from verifiers.v1.assessments import (
    Assessment,
    AssessmentBatch,
    AssessmentContext,
    AssessmentRequest,
    Assessor,
    SourceSnapshot,
)
from verifiers.v1.utils.decorators import invoke


def _validate_dependency_visibility(
    request: AssessmentRequest, dependencies: tuple[Assessment, ...]
) -> None:
    # A parent's view_id does not prove what else its producer could read.
    if any(view.scope == "prefix" for view in request.views) and dependencies:
        raise ValueError("prefix assessment requires qualified upstream visibility")


@validation_owner(borrow=True)
async def execute_assessment_plan(
    hooks: Mapping[str, Callable[..., Any]],
    requests: list[tuple[str, AssessmentRequest]],
    source: SourceSnapshot,
    retained: list[AssessmentBatch],
    max_concurrent: int,
    dependencies: Mapping[str, tuple[Assessment, ...]] | None = None,
) -> None:
    """Validate an independent plan before any dispatch, then execute it bounded.

    Accepted dependencies are selected by trusted runtime code per attempt; they
    are not supplied by the producer. Derived/gated steps stay inside producers.
    """
    if max_concurrent < 1:
        raise ValueError("assessment concurrency must be positive")
    source = SourceSnapshot.model_validate(source.model_dump(mode="python"))
    requests = [
        (name, AssessmentRequest.model_validate(request.model_dump(mode="python")))
        for name, request in requests
    ]
    if {name for name, _ in requests} != set(hooks):
        raise ValueError("assessment requests must account for registered hooks")
    for field in ("run_id", "invocation_id", "attempt_id"):
        identities = [getattr(request.run, field) for _, request in requests]
        previous = {getattr(batch.run, field) for batch in retained}
        if len(identities) != len(set(identities)) or previous.intersection(identities):
            raise ValueError(f"duplicate assessment {field}")
    dependencies = dependencies or {}
    attempt_ids = {request.run.attempt_id for _, request in requests}
    if set(dependencies) - attempt_ids:
        raise ValueError("dependencies name an unplanned assessment attempt")
    # Revalidate caller records once; attempts reuse the admitted copies.
    dependencies = {
        attempt: tuple(
            Assessment.model_validate(parent.model_dump(mode="python"))
            for parent in parents
        )
        for attempt, parents in dependencies.items()
    }
    for _, request in requests:
        if request.source != source.identity:
            raise ValueError("assessment request source mismatch")
        _validate_dependency_visibility(
            request, dependencies.get(request.run.attempt_id, ())
        )
        AssessmentBatch.model_validate(
            {
                "source": source,
                "run": request.run.model_dump(mode="python") | {"status": "running"},
                # Views were admitted from dumps above; batch checks still run.
                "views": request.views,
                "dependencies": dependencies.get(request.run.attempt_id, ()),
            }
        )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(name: str, request: AssessmentRequest) -> None:
        def produce(request: AssessmentRequest, ctx: AssessmentContext):
            return invoke(
                hooks[name],
                {
                    "request": request,
                    "ctx": ctx,
                    "context": ctx,
                    AssessmentRequest: request,
                    AssessmentContext: ctx,
                },
            )

        with planned_validation_child():
            await _execute_admitted(
                produce,
                request,
                source,
                retained,
                dependencies=dependencies.get(request.run.attempt_id, ()),
                semaphore=semaphore,
            )

    async with asyncio.TaskGroup() as group:
        for name, request in requests:
            group.create_task(run_one(name, request))


@validation_owner(borrow=True)
async def execute_assessment(
    producer: Callable[..., Any] | Assessor,
    request: AssessmentRequest,
    source: SourceSnapshot,
    retained: list[AssessmentBatch],
    dependencies: tuple[Assessment, ...] = (),
    semaphore: asyncio.Semaphore | None = None,
) -> AssessmentBatch:
    """Append running, progress and terminal records without erasing prior attempts.

    ``dependencies`` is the caller's explicitly accepted upstream evidence. A
    producer cannot substitute its own parent records. Callback/validation errors
    become failed attempts; cancellation retains an interrupted attempt and escapes.
    An interrupted invocation is retried with a new attempt and invocation identity.
    """
    # Revalidate dumps: frozen models can still be forged through model_copy(update=...).
    source = SourceSnapshot.model_validate(source.model_dump(mode="python"))
    request = AssessmentRequest.model_validate(request.model_dump(mode="python"))
    if request.source != source.identity:
        raise ValueError("assessment request does not belong to the supplied source")
    dependencies = tuple(
        Assessment.model_validate(parent.model_dump(mode="python"))
        for parent in dependencies
    )
    _validate_dependency_visibility(request, dependencies)
    return await _execute_admitted(
        producer, request, source, retained, dependencies, semaphore
    )


@validation_owner(borrow=True)
async def _execute_admitted(
    producer: Callable[..., Any] | Assessor,
    request: AssessmentRequest,
    source: SourceSnapshot,
    retained: list[AssessmentBatch],
    dependencies: tuple[Assessment, ...],
    semaphore: asyncio.Semaphore | None,
) -> AssessmentBatch:
    """Run one attempt whose source, request and dependencies this module admitted.

    Callers pass only models revalidated from dumps by ``execute_assessment`` or
    ``execute_assessment_plan``, so lifecycle records reuse those immutable views
    and dependencies instead of re-dumping them; every record still runs the
    contextual ``AssessmentBatch`` checks.
    """
    if any(batch.run.attempt_id == request.run.attempt_id for batch in retained):
        raise ValueError("assessment attempt identity has already been retained")
    context = AssessmentContext(views=request.views, dependencies=dependencies)
    context._sealed_source = source
    collected: list[Assessment] = []

    def validate(
        assessments: tuple[Assessment, ...],
        *,
        status: str,
        reason: str | None = None,
        returned: AssessmentBatch | None = None,
    ) -> AssessmentBatch:
        run_values = request.run.model_dump(mode="python")
        run_values.update(status=status, reason=reason)
        if returned is not None:
            # Reparse even an existing model to check nested identity and validity.
            returned = AssessmentBatch.model_validate(returned.model_dump(mode="python"))
            returned_source = returned.source
            identity = (
                returned_source.identity
                if isinstance(returned_source, SourceSnapshot)
                else returned_source
            )
            if identity != source.identity:
                raise ValueError("producer changed source identity")
            if (
                isinstance(returned_source, SourceSnapshot)
                and returned_source != source
            ):
                raise ValueError("producer changed retained source input")
            if returned.views != request.views:
                raise ValueError("producer changed or added observation views")
            if returned.dependencies != dependencies:
                raise ValueError("producer supplied unaccepted upstream evidence")
            immutable_run = {"status", "reason", "output_evidence"}
            if returned.run.model_dump(exclude=immutable_run) != request.run.model_dump(
                exclude=immutable_run
            ):
                raise ValueError("producer changed requested run or coverage")
            run_values["output_evidence"] = returned.run.model_dump(mode="json")[
                "output_evidence"
            ]
        run_values["execution_evidence"] = [
            receipt.model_dump(mode="json")
            for receipt in (
                *request.run.execution_evidence,
                *context.execution_evidence,
            )
        ]
        return AssessmentBatch.model_validate(
            {
                "source": source,
                "run": run_values,
                "views": request.views,
                "assessments": [item.model_dump(mode="python") for item in assessments],
                "dependencies": dependencies,
            }
        )

    # A native in-memory marker exists before dispatch. Crash durability additionally
    # requires the caller's artifact/wire persistence to observe these journal events.
    retained.append(
        validate((), status="queued" if semaphore is not None else "running")
    )

    def accept_yield(item: Assessment) -> None:
        if not isinstance(item, Assessment):
            raise TypeError("assessors must yield Assessment records")
        candidate = validate((*collected, item), status="partial")
        collected[:] = candidate.assessments
        retained.append(candidate)

    acquired = False
    try:
        if semaphore is not None:
            await semaphore.acquire()
            acquired = True
            retained.append(validate((), status="running"))
        callback = getattr(producer, "assess", producer)
        if not callable(callback):
            raise TypeError("assessment producer must be callable")
        result = callback(request, context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, AssessmentBatch):
            candidate = validate(
                result.assessments,
                status=result.run.status,
                reason=result.run.reason,
                returned=result,
            )
            if candidate.run.status not in {
                "complete",
                "partial",
                "failed",
                "interrupted",
            }:
                raise ValueError("producer returned a nonterminal running status")
        else:
            if isinstance(result, AsyncIterable):
                async for item in result:
                    accept_yield(item)
            elif isinstance(result, Iterable) and not isinstance(
                result, (str, bytes, dict)
            ):
                for item in result:
                    accept_yield(item)
            else:
                raise TypeError("assessor must return a batch or yield assessments")
            candidate = validate(tuple(collected), status="complete")
    except asyncio.CancelledError:
        retained.append(
            validate(
                tuple(collected),
                status="interrupted",
                reason="cancelled_before_dispatch"
                if semaphore is not None and not acquired
                else "cancelled",
            )
        )
        raise
    except Exception as error:  # noqa: BLE001 - failed plugins are retained as evidence.
        # Do not serialize exception text: it can contain credentials or provider inputs.
        candidate = validate(
            tuple(collected),
            status="failed",
            reason=f"producer_error:{type(error).__name__}",
        )
    finally:
        if acquired and semaphore is not None:
            semaphore.release()
    retained.append(candidate)
    return candidate
