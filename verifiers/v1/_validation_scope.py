"""Private, execution-owned reuse of exact successful intrinsic evidence checks."""

from __future__ import annotations

import asyncio
import sys
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel

_MAX_ENTRIES = 64
_MAX_BYTES = 64 * 1024 * 1024
P = ParamSpec("P")
R = TypeVar("R")


def _task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _typed(value: Any) -> tuple:
    """Read original Python field values; never normalize coordinate types."""
    kind = type(value)
    if isinstance(value, BaseModel):
        return (kind, tuple((name, _typed(getattr(value, name))) for name in kind.model_fields))
    if kind in {tuple, list}:
        return (kind, tuple(_typed(item) for item in value))
    if kind is dict:
        return (kind, tuple((_typed(key), _typed(item)) for key, item in value.items()))
    if kind is str and len(value) > 4096:
        # Exact, reversible representation: a single non-BMP character can
        # otherwise make an entire large JSON string use four bytes per codepoint.
        # Keep the original type tag; no coordinate or content normalization.
        return (kind, value.encode("utf-8", errors="surrogatepass"))
    if value is None or kind in {str, int, float, bool}:
        return (kind, value)
    raise ValueError("unsupported intrinsic proof metadata")


def _size(value: tuple) -> int:
    # Account strings and constructed tuple metadata without encoding. Shared
    # values count repeatedly; classes already belong to loaded modules. This
    # bounds accounted proof storage rather than process resident memory.
    return sys.getsizeof(value) + sum(
        _size(item) if type(item) is tuple else 0 if isinstance(item, type) else sys.getsizeof(item)
        for item in value)


@dataclass
class _Owner:
    task: asyncio.Task[Any] | None
    proofs: OrderedDict[tuple, int] = field(default_factory=OrderedDict)
    retained_bytes: int = 0
    closed: bool = False

    def allowed(self) -> bool:
        current = _task()
        grant = _child.get()
        return not self.closed and (current is self.task or (
            grant is not None and grant[0] is self and grant[1] is current))


_owner: ContextVar[_Owner | None] = ContextVar("native_intrinsic_owner", default=None)
_child: ContextVar[tuple[_Owner, asyncio.Task[Any] | None] | None] = ContextVar(
    "native_intrinsic_child", default=None
)


@contextmanager
def validation_scope(*, borrow: bool = False) -> Iterator[None]:
    previous = _owner.get()
    if borrow and previous is not None and previous.allowed():
        yield
        return
    owner = _Owner(_task())
    token = _owner.set(owner)
    try:
        yield
    finally:
        owner.closed = True
        owner.proofs.clear()
        owner.retained_bytes = 0
        _owner.reset(token)


@contextmanager
def planned_validation_child() -> Iterator[None]:
    """Only the executor's exact current child receives a borrowing permission."""
    owner = _owner.get()
    if owner is None or owner.closed:
        yield
        return
    token = _child.set((owner, _task()))
    try:
        yield
    finally:
        _child.reset(token)


def validation_owner(*, borrow: bool = False):
    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with validation_scope(borrow=borrow):
                return await function(*args, **kwargs)
        return wrapped
    return decorate


@dataclass(frozen=True)
class _Proof:
    owner: _Owner | None
    key: tuple | None
    hit: bool = False

    def remember(self) -> None:
        owner, key = self.owner, self.key
        if owner is None or key is None or not owner.allowed():
            return
        size = _size(key) + 256  # Conservative entry bookkeeping allowance.
        if size > _MAX_BYTES:
            return
        if key in owner.proofs:
            owner.proofs.move_to_end(key)
            return
        owner.proofs[key] = size
        owner.retained_bytes += size
        while len(owner.proofs) > _MAX_ENTRIES or owner.retained_bytes > _MAX_BYTES:
            _, removed = owner.proofs.popitem(last=False)
            owner.retained_bytes -= removed


def intrinsic_proof(model: BaseModel) -> _Proof:
    owner = _owner.get()
    if owner is None or not owner.allowed():
        return _Proof(None, None)
    key = _typed(model)
    if key in owner.proofs:
        owner.proofs.move_to_end(key)
        return _Proof(owner, key, True)
    return _Proof(owner, key)
