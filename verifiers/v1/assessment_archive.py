"""Intern sealed sources and views inside one native owner's evidence archive.

Trace and Episode own their respective histories. These helpers do not traverse
child traces, change standalone batch formats, or introduce an external store.
Restoration supplies full validated snapshots to the existing native validators.

In the archive form each full source and inline view appears once per owner, in
``assessment_sources`` and ``assessment_views``. A batch or credit request names
its source by a compact reference (``snapshot_id`` and ``episode_id``) and each
inline view by ``{"archive_view_ref": {view_id, snapshot_id, builder_revision,
scope, input_digest}}``. Both identities are content digests that the pooled
entry is verified against, so the omitted coordinates add no integrity. Older
archives whose references repeat every coordinate still load: any coordinate a
reference supplies must equal the pooled entry's.

An assessment attempt retains its queued, running and progress batches before
its last batch; they are lifecycle evidence that task credit planners and
calibration inspect. The archive keeps every one, but writes an earlier batch of
an attempt as a delta against that attempt's last batch (``LIFECYCLE_OF``):
only run fields that differ, plus the run identity coordinates, and its
assessments as a prefix count when they are a prefix of the last batch's. The
last batch of each attempt stays complete. Restoration rebuilds the exact batch
dictionaries before native validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SerializationInfo, SerializerFunctionWrapHandler

from verifiers.v1.assessments import (
    ARCHIVE_SAME,
    ObservationView,
    SourceSnapshot,
    archive_serialization,
)

ARCHIVE_CONTEXT = "verifiers_assessment_archive"
"""Serialization-context key requesting the pooled archive form in Python mode.

JSON-mode dumps always use the pooled form. A Python-mode caller that ships the
dump elsewhere (the env server's msgpack reply) sets this key to ``True`` so
each full source and view travels once instead of once per batch.
"""


LIFECYCLE_OF = "archive_lifecycle_of"
"""Archive-form key of an earlier lifecycle batch: index of its attempt's last batch."""

_PREFIX = "archive_prefix"
_ABSENT = "archive_absent"
_DELTA_KEYS = {LIFECYCLE_OF, "schema_version", "run", "assessments", _ABSENT}
_ATTEMPT = ("run_id", "invocation_id", "attempt_id")
# Always written on a delta so raw readers can key and order every batch.
_RUN_COORDINATES = (
    "run_id",
    "producer_id",
    "snapshot_id",
    "invocation_id",
    "attempt_id",
    "status",
)
_VIEW_REFERENCE = (
    "view_id",
    "snapshot_id",
    "builder_revision",
    "scope",
    "input_digest",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - surfaced through native model validators
            f"assessment archive {label} must be an object"
        )
    return value


def _same_wire(left: Any, right: Any) -> bool:
    """Compare wire contents without equating bool/int or coercing scalar types.

    Repeated large payloads are canonical strings, so this does not parse or
    clone their bytes. Type checks also prevent a forged model_copy from using
    Python's True == 1 equality to bypass unique-source admission validation.
    """
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_wire(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_wire(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _same_supplied(reference: dict[str, Any], pooled: dict[str, Any]) -> bool:
    """Every coordinate a reference supplies equals the pooled entry's exactly."""
    return reference.keys() <= pooled.keys() and all(
        _same_wire(value, pooled[key]) for key, value in reference.items()
    )


def _provided_fields_match(raw: Any, validated: Any) -> bool:
    """Reject coercion of supplied coordinates, while allowing legacy defaults.

    Python-mode dumps use tuples where JSON has arrays. This representation
    difference is harmless; changing a boolean or string into an index is not.
    """
    if isinstance(raw, dict) and isinstance(validated, dict):
        return raw.keys() <= validated.keys() and all(
            _provided_fields_match(value, validated[key]) for key, value in raw.items()
        )
    if isinstance(raw, (list, tuple)) and isinstance(validated, (list, tuple)):
        return len(raw) == len(validated) and all(
            _provided_fields_match(a, b) for a, b in zip(raw, validated, strict=True)
        )
    return type(raw) is type(validated) and raw == validated


class _Sources:
    def __init__(self) -> None:
        self.raw: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, SourceSnapshot] = {}
        self.identities: dict[str, dict[str, Any]] = {}
        self.references: dict[str, dict[str, Any]] = {}
        self.pool_ids: set[str] = set()
        self.used: set[str] = set()

    def admit(self, raw: dict[str, Any]) -> SourceSnapshot:
        identity = raw.get("snapshot_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("assessment archive source identity is missing")
        if identity in self.snapshots:
            if not _same_wire(raw, self.raw[identity]):
                raise ValueError("assessment archive conflicting full source")
            return self.snapshots[identity]
        # Reparse raw dictionaries, including model dumps, rather than trusting
        # a caller-supplied frozen instance that could have been forged.
        snapshot = SourceSnapshot.model_validate(raw)
        canonical = snapshot.model_dump(mode="json")
        # SourceSnapshot omits an empty execution inventory on the wire, but
        # older inline archives may have supplied that same empty inventory.
        canonical.setdefault("executions", [])
        if not _provided_fields_match(raw, canonical):
            raise ValueError("assessment archive source coordinate types differ")
        self.raw[identity] = raw
        self.snapshots[identity] = snapshot
        self.identities[identity] = snapshot.identity.model_dump(mode="json")
        self.references[identity] = {
            "snapshot_id": snapshot.snapshot_id,
            "episode_id": snapshot.episode_id,
        }
        return snapshot

    def resolve(self, value: Any) -> SourceSnapshot:
        raw = _mapping(value, "source")
        if "source_json" in raw:
            snapshot = self.admit(raw)
        else:
            identity = raw.get("snapshot_id")
            if not isinstance(identity, str) or identity not in self.snapshots:
                raise ValueError("assessment archive dangling source identity")
            # Compact references name the snapshot; older archives repeat its
            # whole identity. Either way, supplied coordinates must match.
            if "snapshot_id" not in raw or not _same_supplied(
                raw, self.identities[identity]
            ):
                raise ValueError("assessment archive source identity contents differ")
            snapshot = self.snapshots[identity]
        self.used.add(snapshot.snapshot_id)
        return snapshot


def _sources(data: dict[str, Any]) -> _Sources:
    sources = _Sources()
    if "assessment_sources" not in data:
        return sources
    pool = data["assessment_sources"]
    if not isinstance(pool, list):
        raise ValueError(  # noqa: TRY004 - surfaced through native model validators
            "assessment archive sources must be a list"
        )
    for value in pool:
        raw = _mapping(value, "pooled source")
        if "source_json" not in raw:
            raise ValueError("assessment archive pool requires full sources")
        snapshot = sources.admit(raw)
        if snapshot.snapshot_id in sources.pool_ids:
            raise ValueError("assessment archive duplicate pooled source")
        sources.pool_ids.add(snapshot.snapshot_id)
    return sources


class _Views:
    def __init__(self) -> None:
        self.raw: dict[str, dict[str, Any]] = {}
        self.views: dict[str, ObservationView] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.references: dict[str, dict[str, Any]] = {}
        self.pool_ids: set[str] = set()
        self.used: set[str] = set()

    def admit(self, raw: dict[str, Any]) -> ObservationView:
        identity = raw.get("view_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("assessment archive view identity is missing")
        if identity in self.views:
            if not _same_wire(raw, self.raw[identity]):
                raise ValueError("assessment archive conflicting full view")
            return self.views[identity]
        view = ObservationView.model_validate(raw)
        canonical = view.model_dump(mode="json")
        if not _provided_fields_match(raw, canonical):
            raise ValueError("assessment archive view coordinate types differ")
        self.raw[identity] = raw
        self.views[identity] = view
        self.metadata[identity] = {
            key: value for key, value in canonical.items() if key != "input_json"
        }
        self.references[identity] = {
            "archive_view_ref": {
                key: canonical[key] for key in _VIEW_REFERENCE if key in canonical
            }
        }
        return view

    def resolve(self, value: Any) -> ObservationView:
        raw = _mapping(value, "view")
        if "archive_view_ref" in raw:
            if set(raw) != {"archive_view_ref"}:
                raise ValueError("assessment archive view reference has extra fields")
            metadata = _mapping(raw["archive_view_ref"], "view reference")
            identity = metadata.get("view_id")
            if not isinstance(identity, str) or identity not in self.views:
                raise ValueError("assessment archive dangling view identity")
            view = self.views[identity]
            if view.input_json is None:
                raise ValueError(
                    "assessment archive view reference requires inline input"
                )
            if not _same_supplied(metadata, self.metadata[identity]):
                raise ValueError("assessment archive view identity contents differ")
        else:
            view = self.admit(raw)
        self.used.add(view.view_id)
        return view


def _views(data: dict[str, Any]) -> _Views:
    views = _Views()
    if "assessment_views" not in data:
        return views
    pool = data["assessment_views"]
    if not isinstance(pool, list):
        raise ValueError(  # noqa: TRY004 - surfaced through native model validators
            "assessment archive views must be a list"
        )
    for value in pool:
        raw = _mapping(value, "pooled view")
        if raw.get("input_json") is None:
            raise ValueError("assessment archive view pool requires inline input")
        view = views.admit(raw)
        if view.view_id in views.pool_ids:
            raise ValueError("assessment archive duplicate pooled view")
        views.pool_ids.add(view.view_id)
    return views


def _placeholder(value: Any, token: str | None, kind: str) -> str | None:
    """The identity named by this scope's own placeholder, else ``None``."""
    if token is None or type(value) is not dict or set(value) != {ARCHIVE_SAME}:
        return None
    mark = value[ARCHIVE_SAME]
    if (
        type(mark) is list
        and len(mark) == 3
        and mark[0] == token
        and mark[1] == kind
        and type(mark[2]) is str
    ):
        return mark[2]
    return None


def _attempt(batch: Any) -> tuple[Any, ...] | None:
    run = batch.get("run") if type(batch) is dict else None
    if type(run) is not dict or any(type(run.get(key)) is not str for key in _ATTEMPT):
        return None
    return tuple(run[key] for key in _ATTEMPT)


def _delta_value(value: Any, last: Any) -> Any:
    """A proper nonempty prefix of the last batch's sequence as its length."""
    if (
        isinstance(value, (list, tuple))
        and type(value) is type(last)
        and 0 < len(value) < len(last)
        and all(_same_wire(a, b) for a, b in zip(value, last, strict=False))
    ):
        return {_PREFIX: len(value)}
    return value


def _lifecycle_delta(
    batch: dict[str, Any], last: dict[str, Any], index: int
) -> dict[str, Any] | None:
    if batch.keys() != last.keys() or any(
        not _same_wire(value, last[key])
        for key, value in batch.items()
        if key not in {"run", "assessments"}
    ):
        return None
    run, last_run = batch["run"], last["run"]
    delta_run = {key: run[key] for key in _RUN_COORDINATES if key in run}
    for key, value in run.items():
        if key not in delta_run and (
            key not in last_run or not _same_wire(value, last_run[key])
        ):
            delta_run[key] = _delta_value(value, last_run.get(key))
    delta: dict[str, Any] = {}
    if "schema_version" in batch:
        delta["schema_version"] = batch["schema_version"]
    delta[LIFECYCLE_OF] = index
    delta["run"] = delta_run
    if "assessments" in batch and not _same_wire(
        batch["assessments"], last["assessments"]
    ):
        delta["assessments"] = _delta_value(batch["assessments"], last["assessments"])
    if absent := [key for key in last_run if key not in run]:
        delta[_ABSENT] = absent
    return delta


def _compact_lifecycle(batches: list[Any]) -> list[Any]:
    """Write each attempt's earlier batches as deltas against its last batch."""
    last: dict[tuple[Any, ...], int] = {}
    for index, batch in enumerate(batches):
        if (attempt := _attempt(batch)) is not None:
            last[attempt] = index
    compact = []
    for index, batch in enumerate(batches):
        attempt = _attempt(batch)
        final = last.get(attempt) if attempt is not None else None
        delta = None
        if final is not None and final != index:
            delta = _lifecycle_delta(batch, batches[final], final)
        compact.append(batch if delta is None else delta)
    return compact


def _expanded_value(value: Any, last: Any, label: str) -> Any:
    if type(value) is not dict:
        return value
    count = value.get(_PREFIX)
    if (
        set(value) != {_PREFIX}
        or type(count) is not int
        or not isinstance(last, (list, tuple))
        or not 0 < count < len(last)
    ):
        raise ValueError(f"assessment archive lifecycle {label} prefix is invalid")
    return last[:count]


def _expand_lifecycle(batches: list[Any] | tuple[Any, ...]) -> list[Any]:
    """Rebuild lifecycle deltas from their attempt's complete last batch."""
    if not any(type(item) is dict and LIFECYCLE_OF in item for item in batches):
        return list(batches)
    expanded = []
    for index, item in enumerate(batches):
        if type(item) is not dict or LIFECYCLE_OF not in item:
            expanded.append(item)
            continue
        final = item[LIFECYCLE_OF]
        if (
            not item.keys() <= _DELTA_KEYS
            or type(final) is not int
            or not index < final < len(batches)
        ):
            raise ValueError("assessment archive lifecycle reference is invalid")
        last = _mapping(batches[final], "lifecycle batch")
        if LIFECYCLE_OF in last:
            raise ValueError("assessment archive lifecycle reference is not complete")
        last_run = _mapping(last.get("run"), "lifecycle run")
        delta_run = item.get("run")
        if type(delta_run) is not dict or any(
            key not in delta_run or not _same_wire(delta_run[key], last_run.get(key))
            for key in _ATTEMPT
        ):
            raise ValueError("assessment archive lifecycle names another attempt")
        if "schema_version" in item and (
            "schema_version" not in last
            or not _same_wire(item["schema_version"], last["schema_version"])
        ):
            raise ValueError("assessment archive lifecycle schema differs")
        run = dict(last_run)
        absent = item.get(_ABSENT, [])
        if type(absent) is not list or any(key not in run for key in absent):
            raise ValueError("assessment archive lifecycle absent fields are invalid")
        for key in absent:
            del run[key]
        for key, value in delta_run.items():
            run[key] = _expanded_value(value, last_run.get(key), "run")
        batch = dict(last)
        batch["run"] = run
        if "assessments" in item:
            batch["assessments"] = _expanded_value(
                item["assessments"], last.get("assessments"), "assessment"
            )
        expanded.append(batch)
    return expanded


def _rewrite(
    data: dict[str, Any], *, restore: bool, token: str | None = None
) -> tuple[dict[str, Any], int]:
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004 - surfaced through native model validators
            "assessment archive owner must be an object"
        )
    sources = _sources(data)
    views = _views(data)
    result = dict(data)
    result.pop("assessment_sources", None)
    result.pop("assessment_views", None)

    resolved = 0

    def source_value(value: Any) -> SourceSnapshot | dict[str, Any]:
        nonlocal resolved
        identity = _placeholder(value, token, "source")
        if identity is not None:
            # Emitted only for an object whose full dump this scope admitted.
            if identity not in sources.snapshots:
                raise ValueError("assessment archive dangling source identity")
            resolved += 1
            sources.used.add(identity)
            if restore:
                return sources.snapshots[identity]
            return sources.references[identity]
        snapshot = sources.resolve(value)
        return snapshot if restore else sources.references[snapshot.snapshot_id]

    def view_value(value: Any) -> ObservationView | dict[str, Any]:
        nonlocal resolved
        identity = _placeholder(value, token, "view")
        if identity is not None:
            if identity not in views.views:
                raise ValueError("assessment archive dangling view identity")
            resolved += 1
            views.used.add(identity)
            view = views.views[identity]
            if restore:
                return view
            if view.input_json is None:
                return view.model_dump(mode="json")
            return views.references[identity]
        view = views.resolve(value)
        if restore:
            return view
        if view.input_json is None:
            # Artifact-backed views are already compact. Keep their native
            # representation and do not invent artifact resolution here.
            return view.model_dump(mode="json")
        return views.references[view.view_id]

    for field in ("assessment_batches", "credit_assignments"):
        if field not in data:
            continue
        history = data[field]
        if not isinstance(history, (list, tuple)):
            raise ValueError(  # noqa: TRY004 - surfaced through native model validators
                f"assessment archive {field} must be a sequence"
            )
        if field == "assessment_batches":
            history = _expand_lifecycle(history)
        rewritten = []
        for value in history:
            item = dict(_mapping(value, field))
            if field == "assessment_batches":
                if "source" not in item:
                    raise ValueError("assessment archive batch source is missing")
                item["source"] = source_value(item["source"])
                if "views" in item:
                    retained_views = item["views"]
                    if not isinstance(retained_views, (list, tuple)):
                        raise ValueError(
                            "assessment archive batch views must be a sequence"
                        )
                    item["views"] = [view_value(view) for view in retained_views]
            else:
                request = dict(_mapping(item.get("request"), "credit request"))
                if "source" not in request:
                    raise ValueError("assessment archive credit source is missing")
                request["source"] = source_value(request["source"])
                item["request"] = request
            rewritten.append(item)
        if field == "assessment_batches" and not restore:
            rewritten = _compact_lifecycle(rewritten)
        result[field] = rewritten
    if sources.pool_ids - sources.used:
        raise ValueError("assessment archive unused pooled source")
    if views.pool_ids - views.used:
        raise ValueError("assessment archive unused pooled view")
    if not restore and sources.snapshots:
        result["assessment_sources"] = [
            snapshot.model_dump(mode="json") for snapshot in sources.snapshots.values()
        ]
    if not restore:
        inline_views = [
            view.model_dump(mode="json")
            for view in views.views.values()
            if view.input_json is not None
        ]
        if inline_views:
            result["assessment_views"] = inline_views
    return result, resolved


def serialize_archive(
    owner: BaseModel, handler: SerializerFunctionWrapHandler, info: SerializationInfo
) -> Any:
    """Serialize a Trace or Episode, pooling its histories when the mode asks for it."""
    context = info.context
    if info.mode != "json" and not (
        isinstance(context, dict) and context.get(ARCHIVE_CONTEXT) is True
    ):
        return handler(owner)
    with archive_serialization() as scope:
        data = handler(owner)
    if not scope.emitted:
        return normalize_history(data)
    try:
        result, resolved = _rewrite(data, restore=False, token=scope.token)
    except ValueError:
        resolved = -1
    if resolved == scope.emitted:
        return result
    # A placeholder outside the pooled histories, or one met before its full
    # dump, is not resolved here: serialize again in full, which also reports
    # any genuine archive conflict exactly as before.
    return normalize_history(handler(owner))


def normalize_history(data: dict[str, Any]) -> dict[str, Any]:
    """Encode this owner's histories with unique inline source and view pools.

    The returned mapping copies history containers, without a JSON deep clone or
    mutation of its input. Payload strings remain shared until the wire writer.
    Already-normalized input is accepted only when all pool references validate.
    """
    return _rewrite(data, restore=False)[0]


def restore_history(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve archive references or legacy inline evidence to validated models.

    Pool fields are consumed here; evidence ownership remains in native history
    models. Repeated references resolve to the same admitted snapshot/view
    objects, and no history or child-trace metadata is discarded.
    """
    return _rewrite(data, restore=True)[0]
