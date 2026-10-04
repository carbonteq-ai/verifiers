from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import hashlib
import hmac
import inspect
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar
from urllib.parse import parse_qs

from pydantic import TypeAdapter, ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.assessments import canonical_json
from verifiers.v1.mcp.execution import (
    EXECUTION_METADATA_KEY,
    MAX_EXECUTION_RECEIPT_BYTES,
    STATE_CONFLICT_HEADER,
    STATE_EXPECTED_REVISION_HEADER,
    STATE_REVISION_HEADER,
    STATE_WRITE_ID_HEADER,
    DispatchMetadata,
    ExecutionCapture,
    StateRevision,
    active_execution,
    active_revision,
)
from verifiers.v1.state import State, StateT, state_cls
from verifiers.v1.utils.generic import concrete_type

if TYPE_CHECKING:
    from httpx import AsyncClient, Response
    from mcp.server.mcpserver import MCPServer

ConfigT = TypeVar("ConfigT", bound=BaseConfig)
logger = logging.getLogger(__name__)

# State calls may cross a tunnel, so allow transient startup failures without hanging forever.
STATE_TIMEOUT = 30.0  # seconds per request
STATE_RETRIES = 4


async def _channel_request(
    method: str,
    url: str,
    secret: str,
    *,
    route: str | None = None,
    content: bytes | None = None,
    client: AsyncClient | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    """Retry tunnel transport errors and 5xx responses, but not invalid 4xx requests."""
    import httpx
    from tenacity import (
        AsyncRetrying,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    def transient(e: BaseException) -> bool:
        return isinstance(e, httpx.TransportError) or (
            isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
        )

    async def request() -> Response:
        manager = (
            contextlib.nullcontext(client)
            if client is not None
            else httpx.AsyncClient(timeout=STATE_TIMEOUT)
        )
        async with manager as request_client:
            headers = {"Authorization": f"Bearer {secret}"}
            headers.update(extra_headers or {})
            if route:
                headers["X-Verifiers-State-Route"] = route
            if content is not None:
                headers["Content-Type"] = "application/json"
            resp = await request_client.request(
                method, url, content=content, headers=headers
            )
            resp.raise_for_status()
            return resp

    return await AsyncRetrying(
        stop=stop_after_attempt(STATE_RETRIES + 1),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_exception(transient),
        reraise=True,
    )(request)


# Shared servers receive the calling rollout's state channel in URL parameters because one
# process cannot carry a single rollout's channel in its environment.
STATE_URL_PARAM = "vf_state_url"
STATE_ROUTE_PARAM = "vf_state_route"
STATE_SIGNATURE_PARAM = "vf_state_signature"


def state_signature(secret: str, url: str, route: str) -> str:
    """Authenticate one exact shared-server state destination."""
    message = f"vf-state-v1\0{url}\0{route}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


# A context variable isolates the state seen by concurrent calls on one shared server.
_call_state: contextvars.ContextVar[State | None] = contextvars.ContextVar(
    "vf_call_state", default=None
)
_request_query_params: contextvars.ContextVar[dict[str, list[str]] | None] = (
    contextvars.ContextVar("vf_request_query_params", default=None)
)
_request_execution: contextvars.ContextVar[DispatchMetadata | None] = (
    contextvars.ContextVar("vf_request_execution", default=None)
)
# The linked call's submitted arguments. The handler receives them after
# argument validation has filled omitted defaults, so a linked receipt records
# these instead: they are what the dispatch authorized.
_request_arguments: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("vf_request_arguments", default=None)
)


async def _execution_metadata_middleware(ctx, call_next):
    """Scope reserved MCP metadata to this request, including cancellation paths."""
    metadata = None
    arguments = None
    if ctx.method == "tools/call" and isinstance(ctx.params, dict):
        meta = ctx.params.get("_meta")
        if isinstance(meta, dict) and EXECUTION_METADATA_KEY in meta:
            metadata = DispatchMetadata.model_validate(meta[EXECUTION_METADATA_KEY])
            submitted = ctx.params.get("arguments")
            arguments = dict(submitted) if isinstance(submitted, dict) else {}
    token = _request_execution.set(metadata)
    arguments_token = _request_arguments.set(arguments)
    try:
        return await call_next(ctx)
    finally:
        _request_arguments.reset(arguments_token)
        _request_execution.reset(token)


def _request_query(name: str) -> str | None:
    params = _request_query_params.get()
    values = params.get(name, []) if params else []
    if len(values) > 1:
        raise ValueError(f"duplicate {name!r} state coordinate")
    return values[0] if values else None


def _die_with_parent() -> None:
    """On Linux, prevent a server from outliving its launcher."""
    import ctypes
    import signal

    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _import_ref(ref: str) -> object:
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class ServerBase(Generic[ConfigT, StateT]):
    CAPTURE_EXECUTIONS: ClassVar[bool] = False
    TOOL_PREFIX: ClassVar[str | None] = ""
    """The empty value falls back to the snake-cased class name. None advertises the server's
    tools bare (no `<server>_` prefix); name collisions across servers are then the taskset
    author's concern."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._state_cls = state_cls(type(self))
        self._state_adapter = TypeAdapter(self._state_cls)
        self._inert_state: StateT = self._state_cls()  # type: ignore[assignment]
        self._state_client: AsyncClient | None = None
        self._exit_stack = contextlib.AsyncExitStack()

    @property
    def state(self) -> StateT:
        current = _call_state.get()
        return current if current is not None else self._inert_state  # type: ignore[return-value]

    def _state_channel(self) -> tuple[str | None, str, str | None]:
        """Use an env channel, or authenticate the exact coordinates of a shared call."""
        url = os.environ.get("VF_STATE_URL")
        secret = os.environ.get("VF_STATE_SECRET", "")
        if url:
            return url, secret, None
        shared_url = _request_query(STATE_URL_PARAM)
        route = _request_query(STATE_ROUTE_PARAM)
        signature = _request_query(STATE_SIGNATURE_PARAM)
        if not any((shared_url, route, signature)):
            return None, "", None
        if not secret or not shared_url or not route or not signature:
            raise ValueError("invalid shared state coordinates")
        if not hmac.compare_digest(
            signature, state_signature(secret, shared_url, route)
        ):
            raise ValueError("invalid shared state coordinates")
        return shared_url, secret, route

    async def _pull_state(self) -> State:
        url, secret, route = self._state_channel()
        if not url:
            return self._state_cls()
        response = await _channel_request(
            "GET", url, secret, route=route, client=self._state_client
        )
        revision = active_revision.get()
        if revision is not None and STATE_REVISION_HEADER in response.headers:
            revision.read = int(response.headers[STATE_REVISION_HEADER])
        try:
            return self._state_adapter.validate_json(response.content)
        except ValidationError as e:
            logger.warning(
                "state pull rejected for %s: %s", self._state_cls.__name__, e
            )
            raise

    async def _push_state(self, before: bytes) -> None:
        url, secret, route = self._state_channel()
        if not url:
            return
        state = _call_state.get()
        current = state if state is not None else self._inert_state
        after = self._state_adapter.dump_json(current)
        if after == before:
            if (revision := active_revision.get()) is not None:
                revision.persistence = "unchanged"
            return
        revision = active_revision.get()
        write_headers = {}
        if revision is not None and revision.read is not None:
            write_headers[STATE_EXPECTED_REVISION_HEADER] = str(revision.read)
        if (capture := active_execution.get()) is not None:
            write_headers[STATE_WRITE_ID_HEADER] = capture.invocation_id
        response = await _channel_request(
            "PUT",
            url,
            secret,
            route=route,
            content=after,
            client=self._state_client,
            extra_headers=write_headers,
        )
        if revision is not None:
            revision.persistence = "unknown"
            if STATE_REVISION_HEADER in response.headers:
                revision.written = int(response.headers[STATE_REVISION_HEADER])
                revision.persistence = "applied"
            if (
                STATE_CONFLICT_HEADER in response.headers
                and revision.written is not None
            ):
                revision.conflict = response.headers[STATE_CONFLICT_HEADER] == "true"

    def execution_capture_enabled(self) -> bool:
        """Override after state pull to select capture without exposing a tool argument."""
        return self.CAPTURE_EXECUTIONS

    async def _emit_execution_receipt(self, receipt) -> None:
        from verifiers.v1.mcp.execution import ToolServerReceipt

        receipt = ToolServerReceipt.model_validate(
            receipt.model_dump(mode="python"), strict=True
        )
        encoded = canonical_json(receipt.model_dump(mode="json")).encode("utf-8")
        if len(encoded) > MAX_EXECUTION_RECEIPT_BYTES:
            raise ValueError("tool execution receipt exceeds bounded transport size")
        url, secret, route = self._state_channel()
        if not url or not url.endswith("/state"):
            raise ValueError(
                "tool execution capture requires an authenticated /state channel"
            )
        response = await _channel_request(
            "POST",
            url[: -len("/state")] + "/tool-execution",
            secret,
            route=route,
            content=encoded,
            client=self._state_client,
        )
        ack = response.json()
        if ack.get("ok") is not True or type(ack.get("receipt_seq")) is not int:
            raise ValueError("tool execution receipt was not acknowledged")

    async def _report_secondary_failure(
        self, capture, phase, primary, *, result=None, state_error=None
    ) -> None:
        """Capture failure is secondary to the original tool/cancellation/state exception."""
        try:
            await self._emit_execution_receipt(
                capture.receipt(
                    phase,
                    result=result,
                    error=primary if phase != "returned" else None,
                    state_error=state_error,
                )
            )
        except BaseException as secondary:  # noqa: BLE001 - preserve the already active primary failure
            primary.add_note(
                f"Secondary tool-execution capture failure: {type(secondary).__name__}: {secondary}"
            )
            logger.warning(
                "tool-execution capture failed while preserving primary %s: %s",
                type(primary).__name__,
                type(secondary).__name__,
            )

    async def _fetch_task(self, state_url: str | None, secret: str):
        """Fetch the rollout task; shared task-agnostic servers have no task channel."""
        if not state_url:
            return None
        task_url = (
            state_url[: -len("/state")] + "/task"
            if state_url.endswith("/state")
            else state_url
        )
        data = (await _channel_request("GET", task_url, secret)).json()
        return _import_ref(data["cls"]).model_validate_json(data["task"])

    async def _setup_task_from_channel(
        self, state_url: str | None, secret: str
    ) -> None:
        task = await self._fetch_task(state_url, secret)
        if task is not None:
            await self.setup_task(task)

    def _with_state(self, fn: Callable) -> Callable:
        """Sync state around one call, isolated by a context variable.

        Updates replace the whole state, so concurrent writes are last-write-wins. Mutations that
        must compose need to run sequentially.
        """

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Base State has no fields and rejects extras, so only subclasses need channel sync.
            sync_state = self._state_cls is not State
            revision = StateRevision()
            revision_token = active_revision.set(revision)
            token = None
            capture_token = None
            try:
                state = await self._pull_state() if sync_state else State()
                token = _call_state.set(state)
                capture = None
                metadata = _request_execution.get()
                if metadata is not None and not self.execution_capture_enabled():
                    raise ValueError("linked tool execution requires capture enabled")
                if self.execution_capture_enabled():
                    submitted = (
                        _request_arguments.get() if metadata is not None else None
                    )
                    capture = ExecutionCapture(
                        fn.__name__,
                        canonical_json(
                            {"args": [], "kwargs": submitted}
                            if submitted is not None
                            else {"args": args, "kwargs": kwargs}
                        ),
                        revision,
                        **(metadata.model_dump() if metadata is not None else {}),
                        server_name=self.server_name if metadata is not None else None,
                    )
                    capture_token = active_execution.set(capture)
                    await self._emit_execution_receipt(capture.receipt("dispatch"))
                before = self._state_adapter.dump_json(state) if sync_state else None
                try:
                    result = fn(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                except asyncio.CancelledError as error:
                    if capture is not None:
                        await self._report_secondary_failure(
                            capture, "interrupted", error
                        )
                    raise
                except Exception as error:
                    if capture is not None:
                        await self._report_secondary_failure(capture, "raised", error)
                    raise
                try:
                    if before is not None:
                        await self._push_state(before)
                except BaseException as error:
                    import httpx

                    revision.persistence = (
                        "failed"
                        if isinstance(error, httpx.HTTPStatusError)
                        and 400 <= error.response.status_code < 500
                        else "unknown"
                    )
                    if capture is not None:
                        await self._report_secondary_failure(
                            capture, "returned", error, result=result, state_error=error
                        )
                    raise
                if capture is not None:
                    await self._emit_execution_receipt(
                        capture.receipt("returned", result=result)
                    )
                return result
            finally:
                if capture_token is not None:
                    active_execution.reset(capture_token)
                if token is not None:
                    _call_state.reset(token)
                active_revision.reset(revision_token)

        # MCPServer must advertise the wrapped tool's parameters, not `*args, **kwargs`.
        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    @property
    def server_name(self) -> str:
        if self.TOOL_PREFIX is None:
            return ""
        return self.TOOL_PREFIX or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

    async def setup(self) -> None:
        """Initialize task-agnostic server state."""

    async def setup_task(self, task) -> None:
        """Initialize per-task state; taskset-scoped servers skip this hook."""

    def register(self, mcp: MCPServer) -> None:
        """Register this server's MCP handlers."""
        raise NotImplementedError

    def _serve(self) -> None:
        import asyncio
        import socket
        from pathlib import Path

        import uvicorn
        from mcp.server.mcpserver import MCPServer

        from verifiers import __version__

        _die_with_parent()
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        # Remote runtimes require their forwarded port; local servers let the OS choose. Report it
        # before setup so an expensive setup does not block port discovery.
        # asyncio enables TCP_NODELAY per connection only for sockets whose proto
        # is IPPROTO_TCP; a proto-0 listener leaves Nagle on, and each response
        # (headers, then body) then waits ~40 ms for the client's delayed ACK.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(os.environ.get("MCP_PORT", "0"))))
        port_file = os.environ.get("MCP_PORT_FILE")
        if port_file:
            Path(port_file).write_text(str(sock.getsockname()[1]))

        async def serve() -> None:
            import httpx
            from mcp.server.transport_security import TransportSecuritySettings

            try:
                async with (
                    self._exit_stack,
                    httpx.AsyncClient(timeout=STATE_TIMEOUT) as client,
                ):
                    self._state_client = client
                    await self.setup()
                    state_url, secret, _ = self._state_channel()
                    await self._setup_task_from_channel(state_url, secret)
                    # These servers are reached through localhost or a tunnel, never a browser.
                    security = TransportSecuritySettings(
                        enable_dns_rebinding_protection=False
                    )
                    mcp = MCPServer(
                        self.server_name or type(self).__name__,
                        version=__version__,
                        middleware=[_execution_metadata_middleware],
                    )
                    self.register(mcp)
                    # Modern HTTP is inherently sessionless. Keep the legacy leg stateless too so
                    # older agent clients can use the same reconnect-per-call endpoint.
                    mcp_app = mcp.streamable_http_app(
                        json_response=True,
                        stateless_http=True,
                        transport_security=security,
                    )

                    async def app(scope, receive, send):
                        query = (
                            parse_qs(
                                scope.get("query_string", b"").decode(),
                                keep_blank_values=True,
                            )
                            if scope["type"] == "http"
                            else None
                        )
                        token = _request_query_params.set(query)
                        try:
                            await mcp_app(scope, receive, send)
                        finally:
                            _request_query_params.reset(token)

                    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
                    await server.serve(sockets=[sock])
            finally:
                self._state_client = None

        asyncio.run(serve())

    @classmethod
    def _config_cls(cls) -> type[BaseConfig]:
        """Resolve the server's config specialization through its MRO."""
        if config_cls := concrete_type(cls, BaseConfig):
            return config_cls
        raise TypeError(
            f"{cls.__name__} must parameterize its config, e.g. Toolset[MyConfig]"
        )

    @classmethod
    def run(cls) -> None:
        config_cls = cls._config_cls()
        if "VF_CONFIG" in os.environ:
            config = config_cls.model_validate_json(os.environ["VF_CONFIG"])
        else:
            from pydantic_config import cli

            config = cli(config_cls)
        cls(config)._serve()
