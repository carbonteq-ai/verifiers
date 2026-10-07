"""Intern sealed sources and views inside one native owner's evidence archive.

Trace and Episode own their respective histories. These helpers do not traverse
child traces, change standalone batch formats, or introduce an external store.
Restoration supplies full validated snapshots to the existing native validators.
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
        return snapshot

    def resolve(self, value: Any) -> SourceSnapshot:
        raw = _mapping(value, "source")
        if "source_json" in raw:
            snapshot = self.admit(raw)
        else:
            identity = raw.get("snapshot_id")
            if not isinstance(identity, str) or identity not in self.snapshots:
                raise ValueError("assessment archive dangling source identity")
            if not _same_wire(raw, self.identities[identity]):
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
            if not _same_wire(metadata, self.metadata[identity]):
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
            return sources.identities[identity]
        snapshot = sources.resolve(value)
        return snapshot if restore else sources.identities[snapshot.snapshot_id]

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
            return {"archive_view_ref": views.metadata[identity]}
        view = views.resolve(value)
        if restore:
            return view
        if view.input_json is None:
            # Artifact-backed views are already compact. Keep their native
            # representation and do not invent artifact resolution here.
            return view.model_dump(mode="json")
        return {"archive_view_ref": views.metadata[view.view_id]}

    for field in ("assessment_batches", "credit_assignments"):
        if field not in data:
            continue
        history = data[field]
        if not isinstance(history, (list, tuple)):
            raise ValueError(  # noqa: TRY004 - surfaced through native model validators
                f"assessment archive {field} must be a sequence"
            )
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
