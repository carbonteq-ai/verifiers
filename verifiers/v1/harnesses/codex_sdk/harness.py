"""Isolated trusted SDK controller with no solver execution environment."""

import asyncio
import contextlib
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from pydantic import Field, model_validator

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.types import TextContentPart
from verifiers.v1.utils.aio import run_shielded

SOURCE = Path(__file__).with_name("worker.py").read_text()
MODEL_CATALOG = json.loads(Path(__file__).with_name("model_catalog.json").read_text())


class CodexSdkHarnessConfig(HarnessConfig):
    qualification_endpoint: str | None = None
    """Unauthenticated loopback endpoint for no-inference qualification."""
    auth_file: str | None = None
    """Signed-in Codex auth file, readable only by the trusted controller."""
    approved_mcp_tools: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Named tools preauthorized only on the task's explicitly supplied MCP servers."""
    timeout: float = Field(default=120, gt=0)
    output_budget: int = Field(default=16384, gt=0, strict=True)
    """Per-task cumulative SDK output event threshold, with possible overshoot."""

    @model_validator(mode="after")
    def one_provider(self):
        if (self.auth_file is None) == (self.qualification_endpoint is None):
            raise ValueError("choose exactly one auth_file or qualification_endpoint")
        return self


class CodexSdkHarness(Harness[CodexSdkHarnessConfig]):
    SUPPORTS_MCP = True
    # This process is a trusted credential/controller plane. SDK turns explicitly
    # select no execution environments; the qualified startup profile exposes
    # task MCP plus resource administration only. Toolsets retain their own
    # independent code/container requirements.
    NEEDS_CONTAINER = False
    EXECUTES_CODE = False

    async def setup(self, runtime) -> None:
        if self.config.env or self.config.forward_env or self.config.skills:
            raise HarnessError(
                "SDK qualification refuses inherited environment or skills"
            )
        await runtime.prepare_uv_script(SOURCE, {"UV_FROZEN": "false"}, activate=False)

    async def session(
        self,
        ctx,
        trace,
        runtime,
        endpoint,
        secret,
        mcp_urls,
        data,
        tool_interception_url=None,
    ):
        if not runtime.supports_live_processes:
            raise HarnessError("Codex SDK requires live process support")
        return CodexSdkSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        raise HarnessError("Codex SDK uses its event-ingestion session transport")


def public_prompt(data):
    systems = [] if data.system_prompt is None else [data.system_prompt]
    if isinstance(data.prompt, str):
        return "\n\n".join(systems), [{"type": "text", "text": data.prompt}]
    users = []
    for message in data.prompt or []:
        if message.role == "system" and not users:
            if isinstance(message.content, str):
                systems.append(message.content)
            elif all(isinstance(part, TextContentPart) for part in message.content):
                systems.append("".join(part.text for part in message.content))
            else:
                raise HarnessError("SDK system messages must be textual")
        elif message.role == "user":
            if not isinstance(message.content, str):
                raise HarnessError(
                    "SDK qualification currently supports textual users only"
                )
            users.append({"type": "text", "text": message.content})
        else:
            raise HarnessError(
                "SDK initial prompt rejects history and late system messages"
            )
    if not users:
        raise HarnessError("SDK requires a public user prompt")
    return "\n\n".join(systems), users


class CodexSdkSession(HarnessSession):
    async def _run(self, messages):
        if getattr(self, "_attempted", False):
            raise HarnessError("SDK session cannot overwrite an earlier task attempt")
        self._attempted = True
        if messages is not None:
            raise HarnessError(
                "SDK qualification currently supports initial task attempts only"
            )
        harness = cast(CodexSdkHarness, self.harness)
        system, inputs = public_prompt(self.data)
        aliases = {
            name: name
            for name in sorted(self.mcp_urls)
            if re.fullmatch(r"[a-zA-Z0-9_-]+", name)
        }
        reserved = set(aliases.values())
        for name in sorted(set(self.mcp_urls) - set(aliases)):
            base = f"vf_mcp_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
            alias, ordinal = base, 1
            while alias in reserved:
                alias, ordinal = f"{base}_{ordinal}", ordinal + 1
            aliases[name] = alias
            reserved.add(alias)
        request = {
            "qualification_endpoint": harness.config.qualification_endpoint,
            "auth_file": harness.config.auth_file,
            "approved_mcp_tools": {
                aliases[name]: tools
                for name, tools in harness.config.approved_mcp_tools.items()
                if name in aliases
            },
            "model": self.ctx.model,
            "model_catalog": MODEL_CATALOG,
            "input": inputs,
            "system_prompt": system,
            "mcp_urls": {aliases[name]: url for name, url in self.mcp_urls.items()},
            "tool_timeout": harness.config.tool_timeout,
            "timeout": harness.config.timeout,
            "output_budget": harness.config.output_budget,
        }
        if set(harness.config.approved_mcp_tools) - set(aliases):
            raise HarnessError(
                "SDK approval policy references an unknown task MCP server"
            )
        self.trace.info["codex_sdk"] = {
            "sdk_version": "0.160.0",
            "provider_mode": "signed_in"
            if harness.config.auth_file
            else "qualification",
            "fresh_thread_requested": True,
            "output_budget": harness.config.output_budget,
            "server_aliases": aliases,
            "approved_mcp_tools": request["approved_mcp_tools"],
            "model_request_boundaries": "unavailable",
            "token_alignment": "unavailable",
            "mcp_item_execution_join": "unqualified",
            "output_budget_guarantee": "event_threshold_with_possible_overshoot",
            "events": [],
        }
        journal = self.trace.info["codex_sdk"]["events"]
        program = await self.runtime.prepare_uv_script(
            SOURCE, {"UV_FROZEN": "false"}, activate=False
        )
        process = await self.runtime.open_process(program, {})

        async def drain_stderr():
            async for _ in process.stderr:
                pass  # SDK/account transport stderr is not public task evidence.

        stderr = asyncio.create_task(drain_stderr())
        finished = None
        buffer = bytearray()
        try:
            await process.write((json.dumps(request) + "\n").encode())
            async with asyncio.timeout(harness.config.timeout + 15):
                async for chunk in process.stdout:
                    buffer.extend(chunk)
                    if len(buffer) > 16 * 1024 * 1024:
                        raise HarnessError(
                            "SDK event exceeds retained event size bound"
                        )
                    while b"\n" in buffer:
                        line, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        event = json.loads(line)
                        journal.append(event)
                        if (
                            event["kind"] == "event"
                            and event["event"]["method"] == "item/completed"
                        ):
                            item = event["event"]["params"]["item"]
                            if item["type"] == "agentMessage":
                                self.trace.root_reply = item["text"]
                        elif event["kind"] == "finished":
                            finished = event
                exit_code = await process.wait()
            if buffer:
                raise HarnessError("SDK event stream ended mid-record")
            if exit_code or not finished or not finished["ok"]:
                raise HarnessError(
                    "SDK attempt failed; retained partial events remain available"
                )
            return ProgramResult(
                exit_code=0, stdout=self.trace.root_reply or "", stderr=""
            )
        finally:
            self.trace.state.artifacts["codex_sdk/events.json"] = json.dumps(
                self.trace.info["codex_sdk"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()

            async def stop():
                with contextlib.suppress(Exception):
                    await process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    await process.kill()
                    await process.wait()
                await stderr

            await run_shielded(stop())
