"""Revalidate request-bound native server parents; never join by content/order."""

import json

from pydantic import BaseModel

from verifiers.v1.assessments import canonical_json
from verifiers.v1.interception.tool import ToolHookRequest
from verifiers.v1.types import generated_arguments_equal


def validate_dispatch_call(hook):
    route = hook.mcp_dispatch
    if route is None or hook.call is None:
        raise ValueError("native MCP dispatch destination unavailable")
    qualified = (
        f"{route.server_name}_{route.tool_name}"
        if route.server_name
        else route.tool_name
    )
    arguments = hook.call.arguments
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    if (
        hook.call.name != qualified
        or not isinstance(parsed, dict)
        or canonical_json(parsed) != route.arguments_json
    ):
        raise ValueError("native MCP dispatch differs from original emitted call")


def validate_server_parent(receipt, events, nodes):
    """Check an observed prefix; retries retain independent physical identities."""
    if receipt.parent_execution_id is None:
        return None
    raw = [
        event.model_dump(mode="json") if hasattr(event, "model_dump") else event
        for event in events
    ]
    parents = [
        event
        for event in raw
        if event.get("source") in {"harness", "interceptor"}
        and event.get("execution_id") == receipt.parent_execution_id
        and event.get("phase") == "dispatch"
    ]
    if len(parents) != 1:
        raise ValueError("native server parent dispatch unavailable at observed prefix")
    parent = parents[0]
    hook = ToolHookRequest.model_validate_json(parent["request_json"])
    validate_dispatch_call(hook)
    route = hook.mcp_dispatch
    decision = json.loads(parent["decision_json"])
    if (
        decision.get("action") != "allow"
        or decision.get("mcp_dispatch_ticket") != receipt.dispatch_ticket
    ):
        raise ValueError("native server dispatch ticket was not authorized")
    if (
        route is None
        or receipt.server_name != route.server_name
        or receipt.tool_name != route.tool_name
    ):
        raise ValueError("native server dispatch destination changed")
    actual = json.loads(receipt.arguments_json)
    expected = {"args": [], "kwargs": json.loads(route.arguments_json)}
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("native server dispatch arguments changed")
    index = parent["node_index"]
    if type(index) is not int or not 0 <= index < len(nodes):
        raise ValueError("native server sampled parent node unavailable")
    node = nodes[index]
    sampled = node.get("sampled") if isinstance(node, dict) else node.sampled
    if sampled is not True:
        raise ValueError("native server parent is not a sampled call")
    message = node.get("message") if isinstance(node, dict) else node.message
    message = (
        message.model_dump(mode="json") if isinstance(message, BaseModel) else message
    )
    calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("native server parent is not an assistant call")
    ordinal = parent["emitted_call_index"]
    if type(ordinal) is not int or not 0 <= ordinal < len(calls):
        raise ValueError("native server emitted call ordinal unavailable")
    emitted = calls[ordinal]
    if sum(call.get("id") == emitted.get("id") for call in calls) != 1:
        raise ValueError("native server parent provider ID is ambiguous")
    if hook.call is None:
        raise ValueError("native server original emitted call unavailable")
    submitted = hook.call.model_dump(mode="json")
    if any(
        emitted.get(key) != submitted[key] for key in ("id", "name", "type")
    ) or not generated_arguments_equal(
        emitted.get("arguments"), submitted["arguments"]
    ):
        raise ValueError("native server parent differs from original sampled call")
    if parent.get("generated_attempt_index") is not None:
        attempts = (
            node.get("generated_calls", [])
            if isinstance(node, dict)
            else node.generated_calls
        )
        attempts = [
            attempt.model_dump(mode="json")
            if hasattr(attempt, "model_dump")
            else attempt
            for attempt in attempts
        ]
        if any(
            not isinstance(attempt, dict)
            or type(attempt.get("attempt_index")) is not int
            or (
                attempt.get("emitted_call_index") is not None
                and type(attempt.get("emitted_call_index")) is not int
            )
            for attempt in attempts
        ):
            raise ValueError("native server generated-attempt coordinates are invalid")
        linked = [
            attempt.get("attempt_index")
            for attempt in attempts
            if attempt.get("emitted_call_index") == ordinal
        ]
        if linked != [parent["generated_attempt_index"]]:
            raise ValueError("native server generated-attempt link is not unique")
    for event in raw:
        if event.get("source") != "tool_server":
            continue
        earlier = json.loads(event["receipt_json"])
        if (
            earlier.get("dispatch_ticket") == receipt.dispatch_ticket
            and earlier.get("transport_attempt_index")
            == receipt.transport_attempt_index
            and event["invocation_id"] != receipt.invocation_id
        ):
            raise ValueError(
                "native transport attempt already belongs to another invocation"
            )
    return parent
