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
_MAX_INSTANCES = 4096
P = ParamSpec("P")
R = TypeVar("R")


def _task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


_SCALARS = frozenset({str, int, float, bool, type(None)})
_LARGE_TEXT = 4096


def _typed(value: Any) -> tuple:
    """Read original Python field values; never normalize coordinate types.

    Strings stay in the key as the exact objects the model holds: their hash is
    computed once per object, so repeated lookups of large retained JSON do not
    re-encode or re-hash it, and distinct code points or lone surrogates remain
    distinct keys.
    """
    kind = type(value)
    if kind in _SCALARS:
        return (kind, value)
    if isinstance(value, BaseModel):
        # Stored field values with their names: a missing, extra or reordered
        # field yields a different key (a miss and full validation).
        return (
            kind,
            tuple([(name, _typed(item)) for name, item in value.__dict__.items()]),
        )
    if kind is tuple or kind is list:
        return (kind, tuple([_typed(item) for item in value]))
    if kind is dict:
        return (
            kind,
            tuple([(_typed(key), _typed(item)) for key, item in value.items()]),
        )
    raise ValueError("unsupported intrinsic proof metadata")


def _leaf_size(item: Any) -> int:
    if isinstance(item, type):
        return 0  # Classes already belong to loaded modules.
    if type(item) is str and len(item) > _LARGE_TEXT:
        # Large text is held by reference, normally shared with live models.
        # Account one unit per code point, as compact UTF-8 would for the
        # ASCII-dominant JSON it carries, so astral characters cannot make a
        # single source exhaust the bound.
        return sys.getsizeof("") + len(item)
    return sys.getsizeof(item)


def _size(value: tuple) -> int:
    # Account strings and constructed tuple metadata without encoding. Shared
    # values count repeatedly. This bounds accounted proof storage rather than
    # process resident memory.
    return sys.getsizeof(value) + sum(
        _size(item) if type(item) is tuple else _leaf_size(item) for item in value
    )


def _immutable(key: tuple) -> bool:
    """Whether a typed key describes only frozen models, tuples and scalars."""
    kind, payload = key
    if kind in _SCALARS:
        return True
    if kind is tuple:
        return all(_immutable(item) for item in payload)
    if isinstance(kind, type) and issubclass(kind, BaseModel):
        return kind.model_config.get("frozen") is True and all(
            _immutable(item) for _, item in payload
        )
    return False


@dataclass
class _Owner:
    task: asyncio.Task[Any] | None
    # key -> (accounted size, whether the proven value is deeply immutable)
    proofs: OrderedDict[tuple, tuple[int, bool]] = field(default_factory=OrderedDict)
    # id -> (exact proven object, its proof key); ids stay unique while held.
    instances: OrderedDict[int, tuple[BaseModel, tuple]] = field(
        default_factory=OrderedDict
    )
    retained_bytes: int = 0
    closed: bool = False

    def remember_instance(self, model: BaseModel, key: tuple) -> None:
        self.instances[id(model)] = (model, key)
        self.instances.move_to_end(id(model))
        while len(self.instances) > _MAX_INSTANCES:
            self.instances.popitem(last=False)

    def allowed(self) -> bool:
        current = _task()
        grant = _child.get()
        return not self.closed and (
            current is self.task
            or (grant is not None and grant[0] is self and grant[1] is current)
        )


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
        owner.instances.clear()
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
    model: BaseModel | None = None

    def remember(self) -> None:
        owner, key = self.owner, self.key
        if owner is None or key is None or not owner.allowed():
            return
        size = _size(key) + 256  # Conservative entry bookkeeping allowance.
        if size > _MAX_BYTES:
            return
        immutable = _immutable(key)
        if immutable and self.model is not None:
            owner.remember_instance(self.model, key)
        if key in owner.proofs:
            owner.proofs.move_to_end(key)
            return
        owner.proofs[key] = (size, immutable)
        owner.retained_bytes += size
        while len(owner.proofs) > _MAX_ENTRIES or owner.retained_bytes > _MAX_BYTES:
            _, (removed, _) = owner.proofs.popitem(last=False)
            owner.retained_bytes -= removed


def proven_instance(model: BaseModel) -> bool:
    """Whether this exact object already passed its intrinsic checks in scope.

    Only deeply immutable objects (frozen models, tuples and scalars) are
    recorded, so the object cannot have changed since. A ``model_copy`` or a
    re-parsed copy is a different object and is checked on its own; this
    shortcut also skips the strict coordinate re-admission that object passed.
    """
    owner = _owner.get()
    if owner is None or not owner.allowed():
        return False
    entry = owner.instances.get(id(model))
    if entry is None or entry[0] is not model:
        return False
    # The identity shortcut lasts only while its content proof is retained, so
    # proof eviction still forces full validation.
    if entry[1] not in owner.proofs:
        del owner.instances[id(model)]
        return False
    owner.instances.move_to_end(id(model))
    owner.proofs.move_to_end(entry[1])
    return True


def intrinsic_proof(model: BaseModel) -> _Proof:
    owner = _owner.get()
    if owner is None or not owner.allowed():
        return _Proof(None, None)
    key = _typed(model)
    entry = owner.proofs.get(key)
    if entry is not None:
        owner.proofs.move_to_end(key)
        if entry[1]:
            owner.remember_instance(model, key)
        return _Proof(owner, key, True, model)
    return _Proof(owner, key, model=model)
