# /// script
# requires-python = ">=3.11"
# dependencies = ["openai-codex==0.160.0"]
# ///
"""Isolated trusted SDK controller; credentials are outside the solver environment.

The solver has no execution environment. SDK agent-turn events are retained as
events, not represented as individual model requests or token coordinates.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

SDK_VERSION = "0.160.0"
OUTPUT_SCHEMA_FORWARDING = "codex-turn-output-schema@1"
BUILTIN_TOOL_ISOLATION = "codex-declared-tools-only@1"
DISABLED_BUILTIN_FEATURES = dict.fromkeys(
    (
        "image_generation",
        "browser_use",
        "browser_use_external",
        "browser_use_full_cdp_access",
        "computer_use",
        "in_app_browser",
        "shell_tool",
        "view_image",
        "js_repl",
        "apps",
        "plugins",
        "multi_agent",
        "sleep_tool",
        "goals",
    ),
    False,
)
CATALOG_SOURCE_REVISION = "a956835d020762cb2b570053af06f643a11c0ecc"


def restricted_catalog(catalog: dict, model: str) -> tuple[dict, dict]:
    """Keep the pinned model descriptor; narrow three tool capability fields.

    Models select tool mode and experimental utilities before ordinary feature
    flags in SDK 0.160. The explicit static catalog is a tool policy override,
    not evidence that an authenticated account can access this model.
    """
    original = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
    copied = json.loads(original)
    if len(copied.get("models", [])) != 1 or copied["models"][0]["slug"] != model:
        raise ValueError("SDK capability catalog must match the requested model")
    descriptor = copied["models"][0]
    changes = {
        "tool_mode": "direct",
        "multi_agent_version": "disabled",
        "experimental_supported_tools": [],
    }
    previous = {key: descriptor.get(key) for key in changes}
    descriptor.update(changes)
    restricted = json.dumps(copied, sort_keys=True, separators=(",", ":"))
    return copied, {
        "source_revision": CATALOG_SOURCE_REVISION,
        "source_path": "codex-rs/models-manager/models.json",
        "original_digest": hashlib.sha256(original.encode()).hexdigest(),
        "restricted_digest": hashlib.sha256(restricted.encode()).hexdigest(),
        "previous": previous,
        "changes": changes,
    }


def emit(kind: str, **payload) -> None:
    print(json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def write_catalog(directory: str, catalog: dict) -> str:
    path = os.path.join(directory, "model_catalog.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(catalog, stream)
    return path


def controller_config(request: dict, directory: str) -> dict:
    endpoint = request.get("qualification_endpoint")
    if endpoint is not None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "qualification provider must be unauthenticated loopback HTTP"
            )
    servers = {
        name: {
            "url": url,
            "required": True,
            "tool_timeout_sec": request["tool_timeout"],
            "tools": {
                tool: {"approval_mode": "approve"}
                for tool in request.get("approved_mcp_tools", {}).get(name, ())
            },
        }
        for name, url in request["mcp_urls"].items()
    }
    if set(request.get("approved_mcp_tools", {})) - set(servers):
        raise ValueError("approval policy references an unknown task MCP server")
    bases = {
        name: (namespace if namespace.startswith("mcp__") else f"mcp__{namespace}")
        for name in servers
        for namespace in (re.sub(r"[^a-zA-Z0-9_]", "_", name),)
    }
    counts = Counter(bases.values())
    namespaces = []
    for name, namespace in bases.items():
        if counts[namespace] > 1:
            suffix = hashlib.sha1(f"{name}\0{name}\0".encode()).hexdigest()[:12]
            namespace = (
                f"{namespace[:-2]}_{suffix}__"
                if namespace.endswith("__")
                else f"{namespace}_{suffix}"
            )
        namespaces.append(namespace)
        if len(namespace) > 49:
            namespaces.append(namespace[:49])
    config = {
        "model_provider": "qualification" if endpoint else "openai",
        "forced_login_method": "chatgpt",
        "cli_auth_credentials_store": "file",
        "mcp_servers": servers,
        "agents": {"enabled": False},
        "tools": {"experimental_request_user_input": {"enabled": False}},
        "web_search": "disabled",
        "features": {
            "mcp_2026_07_28": True,
            "code_mode": {
                "direct_only_tool_namespaces": list(dict.fromkeys(namespaces))
            },
            **DISABLED_BUILTIN_FEATURES,
            "multi_agent_v2": {"enabled": False},
        },
        "cwd": directory,
    }
    if endpoint:
        config["model_providers"] = {
            "qualification": {
                "name": "Local qualification fixture",
                "base_url": endpoint,
                "wire_api": "responses",
                "requires_openai_auth": False,
            }
        }
    return config


async def confirm_builtin_tool_isolation(client, thread_id, response_model) -> None:
    """Confirm the loaded thread's refreshed configuration before any turn."""
    observed = {}
    seen_names = set()
    seen_cursors = set()
    cursor = None
    for _ in range(32):
        response = await client.request(
            "experimentalFeature/list",
            {"threadId": thread_id, "limit": 100, "cursor": cursor},
            response_model=response_model,
        )
        payload = response.root
        if type(payload) is not dict or type(payload.get("data")) is not list:
            raise ValueError("SDK builtin feature readback malformed")
        for item in payload["data"]:
            if type(item) is not dict or type(item.get("name")) is not str:
                raise ValueError("SDK builtin feature entry malformed")
            name = item["name"]
            if name in seen_names:
                raise ValueError("SDK builtin feature readback duplicated")
            seen_names.add(name)
            if name in DISABLED_BUILTIN_FEATURES:
                if type(item.get("enabled")) is not bool:
                    raise ValueError("SDK builtin feature enablement malformed")
                observed[name] = item["enabled"]
        if "nextCursor" not in payload:
            raise ValueError("SDK builtin feature pagination unavailable")
        cursor = payload["nextCursor"]
        if cursor is None:
            break
        if type(cursor) is not str or not cursor or cursor in seen_cursors:
            raise ValueError("SDK builtin feature cursor invalid")
        seen_cursors.add(cursor)
    else:
        raise ValueError("SDK builtin feature pagination limit exceeded")
    if observed != DISABLED_BUILTIN_FEATURES:
        raise ValueError("SDK builtin tool isolation not confirmed")
    emit(
        "builtin_tool_isolation",
        revision=BUILTIN_TOOL_ISOLATION,
        scope="loaded_thread",
        thread_id=thread_id,
        disabled_features=observed,
    )


async def run_qualification(request: dict, directory: str, client_factory=None) -> None:
    # Preserve the caller's optional JSON Schema through the public SDK API.
    # Reject malformed envelopes before copying credentials or starting a client.
    turn_params: dict[str, Any] = {"environments": []}
    if "output_schema" in request:
        if type(request["output_schema"]) is not dict:
            raise ValueError("SDK output_schema must be a JSON Schema object")
        turn_params["outputSchema"] = request["output_schema"]
    # CodexConfig.env merges its parent environment, so scrub that parent before
    # creating the SDK/app-server. No host auth/config file is reused.
    clean = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    clean.update(
        HOME=directory,
        CODEX_HOME=f"{directory}/codex",
        XDG_CONFIG_HOME=f"{directory}/config",
        XDG_DATA_HOME=f"{directory}/data",
    )
    for key in ("CODEX_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        os.makedirs(clean[key], mode=0o700, exist_ok=True)
    auth_file = request.get("auth_file")
    if (auth_file is None) == (request.get("qualification_endpoint") is None):
        raise ValueError(
            "choose exactly one signed-in auth file or qualification endpoint"
        )
    if auth_file:
        destination = os.path.join(clean["CODEX_HOME"], "auth.json")
        shutil.copyfile(auth_file, destination)
        os.chmod(destination, 0o600)
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(clean)
    try:
        from openai_codex import CodexConfig
        from openai_codex.async_client import AsyncCodexClient
        from openai_codex.errors import InvalidRequestError
        from pydantic import RootModel

        class RawResponse(RootModel[dict]):
            pass

        config = controller_config(request, directory)
        catalog, provenance = restricted_catalog(
            request["model_catalog"], request["model"]
        )
        config["model_catalog_json"] = write_catalog(directory, catalog)
        emit("capability_catalog", **provenance)
        factory = client_factory or AsyncCodexClient
        startup = ('forced_login_method="chatgpt"', 'cli_auth_credentials_store="file"')
        if auth_file:
            # Discover account/model availability in an unmodified catalog.
            # A static restricted catalog cannot establish account entitlement.
            discovery = CodexConfig(
                cwd=directory,
                env=clean,
                experimental_api=True,
                config_overrides=startup,
            )
            async with factory(discovery) as client:
                await client.initialize()
                account = await client.account_read()
                if account.account is None or account.account.root.type != "chatgpt":
                    raise ValueError(
                        "SDK requires a signed-in ChatGPT account; no API fallback"
                    )
                available = await client.model_list(include_hidden=True)
                if not any(item.model == request["model"] for item in available.data):
                    raise ValueError(
                        "requested SDK model absent from authenticated catalog"
                    )
                emit("authenticated", account_type="chatgpt", model=request["model"])
        # ModelsManager is created at app-server startup; a per-thread override
        # arrives too late to replace its bundled model capabilities.
        sdk_config = CodexConfig(
            cwd=directory,
            env=clean,
            experimental_api=True,
            config_overrides=(
                *startup,
                "model_catalog_json=" + json.dumps(config["model_catalog_json"]),
                *(f"features.{name}=false" for name in DISABLED_BUILTIN_FEATURES),
            ),
        )
        async with factory(sdk_config) as client:
            initialization = await client.initialize()
            emit(
                "initialized",
                sdk_version=SDK_VERSION,
                response=initialization.model_dump(mode="json", by_alias=True),
            )
            started = await client.request(
                "thread/start",
                {
                    "model": request["model"],
                    "modelProvider": "qualification" if not auth_file else "openai",
                    "ephemeral": True,
                    "experimentalRawEvents": True,
                    "cwd": directory,
                    "environments": [],
                    "approvalPolicy": "never",
                    "baseInstructions": request.get("system_prompt") or "",
                    "config": config,
                },
                response_model=RawResponse,
            )
            thread = started.root["thread"]
            if thread.get("environments") != []:
                raise ValueError(
                    "SDK thread did not confirm an explicit empty environment selection"
                )
            thread_id = thread["id"]
            emit(
                "thread_started",
                response=started.root,
                raw_response_events_requested=True,
            )
            await confirm_builtin_tool_isolation(client, thread_id, RawResponse)
            servers = []
            for name in sorted(request["mcp_urls"]):
                inventory = await client.request(
                    "mcpServerStatus/list",
                    {"threadId": thread_id, "serverName": name, "detail": "full"},
                    response_model=RawResponse,
                )
                servers.extend(inventory.root["data"])
            # Controller diagnostic distinguishes discovery from model exposure;
            # omit endpoint/auth metadata and retain only public tool identities.
            emit(
                "mcp_inventory",
                scope="thread_bound",
                servers=[
                    {
                        "name": server["name"],
                        "tool_names": sorted(server.get("tools", {})),
                        "runtime_status": server.get("runtimeStatus"),
                    }
                    for server in servers
                ],
            )
            # Public turn_start buffers events before dispatch; a raw request
            # would discard per-turn events before a subscriber is attached.
            turn = await client.turn_start(thread_id, request["input"], turn_params)
            turn_id = turn.turn.id
            emit("turn_started", response=turn.model_dump(mode="json", by_alias=True))
            budget_interrupted = False
            previous_output = None
            provider_errors_observed = 0
            provider_retries_observed = 0
            try:
                async with asyncio.timeout(request["timeout"]):
                    while True:
                        event = await client.next_turn_notification(turn_id)
                        if not (
                            event.method.startswith(("item/", "turn/", "thread/"))
                            or event.method in {"rawResponse/completed", "error"}
                        ):
                            continue
                        if event.method == "rawResponse/completed" and not hasattr(
                            event.payload, "model_dump"
                        ):
                            # The published 0.160 SDK exposes this experimental
                            # notification as public UnknownNotification.params.
                            params = event.payload.params
                            if not isinstance(params, dict):
                                raise TypeError(
                                    "SDK raw response notification must contain JSON params"
                                )
                        else:
                            params = event.payload.model_dump(
                                mode="json", by_alias=True
                            )
                        raw = {
                            "method": event.method,
                            "params": params,
                        }
                        params = raw["params"]
                        if params.get("threadId") != thread_id or (
                            params.get("turnId") is not None
                            and params["turnId"] != turn_id
                        ):
                            continue  # account/auth/setup events are controller-private.
                        emit("event", event=raw)
                        if event.method == "error":
                            provider_errors_observed += 1
                            provider_retries_observed += int(
                                params.get("willRetry") is True
                            )
                        if event.method == "thread/tokenUsage/updated":
                            total = params["tokenUsage"]["total"]
                            observed_output = total["outputTokens"]
                            if observed_output < 0 or (
                                previous_output is not None
                                and observed_output < previous_output
                            ):
                                emit(
                                    "usage_unusable",
                                    reason="output_counter_regressed_or_negative",
                                    observed_usage=total,
                                )
                                raise ValueError(
                                    "SDK output counter cannot support this attempt budget"
                                )
                            previous_output = observed_output
                            if (
                                not budget_interrupted
                                and observed_output >= request["output_budget"]
                            ):
                                budget_interrupted = True
                                interrupt_result = "accepted"
                                try:
                                    await client.request(
                                        "turn/interrupt",
                                        {"threadId": thread_id, "turnId": turn_id},
                                        response_model=RawResponse,
                                    )
                                except InvalidRequestError as interruption:
                                    if (
                                        interruption.code != -32600
                                        or interruption.message
                                        != "no active turn to interrupt"
                                    ):
                                        raise
                                    # Usage can precede an already queued terminal
                                    # event after the runtime has finished the turn.
                                    # Await that event; this response alone does not
                                    # establish completion or erase excess output.
                                    interrupt_result = "inactive_awaiting_terminal"
                                emit(
                                    "budget_interrupt",
                                    interrupt_result=interrupt_result,
                                    observed_usage=total,
                                    threshold=request["output_budget"],
                                    guarantee="event_threshold_with_possible_overshoot",
                                    reasoning_inclusion="unqualified",
                                )
                        if (
                            event.method == "turn/completed"
                            and raw["params"]["turn"]["id"] == turn_id
                        ):
                            status = raw["params"]["turn"]["status"]
                            emit(
                                "finished",
                                ok=status == "completed",
                                thread_id=thread_id,
                                turn_id=turn_id,
                                status=status,
                                provider_errors_observed=provider_errors_observed,
                                provider_retries_observed=provider_retries_observed,
                                output_budget_state="unqualified_provider_errors_observed"
                                if provider_errors_observed
                                else "unavailable"
                                if previous_output is None
                                else "observed_output_only",
                            )
                            return
            except BaseException as error:
                try:
                    await client.request(
                        "turn/interrupt",
                        {"threadId": thread_id, "turnId": turn_id},
                        response_model=RawResponse,
                    )
                except Exception as secondary:  # noqa: BLE001 - preserve primary failure
                    error.add_note(
                        f"Secondary SDK interruption failure: {type(secondary).__name__}"
                    )
                emit("interrupted", thread_id=thread_id, turn_id=turn_id)
                raise
    finally:
        os.environ.clear()
        os.environ.update(previous)


def main() -> None:
    request = json.loads(sys.stdin.readline())
    try:
        with tempfile.TemporaryDirectory(prefix="vf-codex-sdk-") as directory:
            asyncio.run(run_qualification(request, directory))
    except Exception as error:  # noqa: BLE001 - worker protocol retains terminal failure
        emit(
            "error",
            error_type=type(error).__name__,
            detail="Signed-in SDK controller failed; private diagnostic omitted"
            if request.get("auth_file")
            else str(error),
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
