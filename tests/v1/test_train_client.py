import json
from types import SimpleNamespace

import pytest
from renderers import DefaultRendererConfig

from verifiers.v1.clients.renderer_extensions import (
    K2IFMReasoningParser,
    K2IFMToolParser,
    LFM2ToolParser,
    bridge_lfm2_tool_cycle,
)
from verifiers.v1.clients.train import ElasticRendererPool
from verifiers.v1.configs.client import TrainClientConfig


def test_generated_call_attempts_keep_original_ordinals_and_sampled_spans():
    from renderers.base import ParsedToolCall, ToolCallParseStatus

    import verifiers.v1 as vf
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients.train import response_from_generate
    from verifiers.v1.graph import prepare_turn

    producer = vf.GeneratedCallProducer.capture(
        "fixture-original-span-parser@1", {"parser": "fixture", "version": 1}
    )

    completion = list(range(20, 30))
    calls = [
        ParsedToolCall(
            raw="send(1)", name="send", arguments={"id": 1}, token_span=(1, 3)
        ),
        ParsedToolCall(
            raw="broken", token_span=(3, 5), status=ToolCallParseStatus.INVALID_JSON
        ),
        ParsedToolCall(
            raw="unknown()",
            name="unknown",
            token_span=(5, 7),
            status=ToolCallParseStatus.UNKNOWN_TOOL,
        ),
        ParsedToolCall(
            raw="send(1)", name="send", arguments={"id": 1}, token_span=(7, 9)
        ),
    ]
    response = response_from_generate(
        {
            "prompt_ids": [11, 12],
            "completion_ids": completion,
            "completion_logprobs": [-0.1] * len(completion),
            "tool_calls": calls,
        },
        "model",
        parser_revision="fixture-original-span-parser@1",
        producer=producer,
    )
    assert [call.name for call in response.message.tool_calls] == ["send", "send"]
    assert [call.emitted_call_index for call in response.tokens.generated_calls] == [
        0,
        None,
        None,
        1,
    ]
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="send once")),
    )
    node_index = prepare_turn(trace, [vf.UserMessage(content="send once")]).commit(
        response
    )
    node = trace.nodes[node_index]
    assert [call.token_span for call in node.generated_calls] == [
        (3, 5),
        (5, 7),
        (7, 9),
        (9, 11),
    ]
    assert all(
        call.coordinate_system == "node_local_full_tokens"
        for call in node.generated_calls
    )
    source = capture_trace_source(trace)
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert restored.nodes[node_index].generated_calls == node.generated_calls
    assert restored.nodes[node_index].generated_call_producer == producer
    assert json.loads(source.source_json)["nodes"][node_index][
        "generated_call_producer"
    ] == producer.model_dump(mode="json")
    for ordinal, span in enumerate(((3, 5), (5, 7), (7, 9), (9, 11))):
        subject = vf.SubjectRef(
            kind="call",
            snapshot_id=source.snapshot_id,
            episode_id="episode",
            trace_id=trace.id,
            node_index=node_index,
            node_content_digest=source.nodes[node_index].node_content_digest,
            call_index=ordinal,
        )
        projection = vf.project_subject(subject, source, restored)
        assert projection.status == "exact_call"
        assert [(item.start, item.end) for item in projection.intervals] == [span]
        assert all(node.mask[slice(*span)])
    changed = restored.model_copy(deep=True)
    changed.nodes[node_index].generated_calls = tuple(
        call.model_copy(update={"raw": "rewritten"})
        if call.attempt_index == 3
        else call
        for call in changed.nodes[node_index].generated_calls
    )
    assert vf.project_subject(subject, source, changed).status == "failed"
    changed = restored.model_copy(deep=True)
    changed.nodes[
        node_index
    ].generated_call_producer = vf.GeneratedCallProducer.capture(
        producer.parser_revision, {"parser": "different implementation"}
    )
    assert vf.project_subject(subject, source, changed).status == "failed"
    for change in ({"descriptor_digest": "wrong"}, {"parser_revision": "different"}):
        forged = response.model_copy(deep=True)
        forged.tokens.generated_call_producer = producer.model_copy(update=change)
        empty = vf.Trace(episode_id="episode", agent=trace.agent, task=trace.task)
        with pytest.raises(ValueError):
            prepare_turn(empty, [vf.UserMessage(content="request")]).commit(forged)
        assert empty.nodes == []
    with pytest.raises(ValueError):
        vf.GeneratedCallProducer.model_validate(
            {
                **producer.model_dump(),
                "descriptor_json": '{ "parser": "fixture", "version": 1 }',
            }
        )
    record = trace.to_record()
    record["nodes"][node_index]["generated_call_producer"]["descriptor_digest"] = (
        "wrong"
    )
    with pytest.raises(ValueError):
        vf.WireTrace.model_validate(record)
    # Directly constructed/replayed nodes must satisfy the same link rules.
    changed = restored.model_copy(deep=True)
    changed.nodes[node_index].generated_calls = tuple(
        call.model_copy(update={"emitted_call_index": 99})
        if call.attempt_index == 3
        else call
        for call in changed.nodes[node_index].generated_calls
    )
    changed_source = capture_trace_source(changed)
    changed_subject = subject.model_copy(
        update={
            "snapshot_id": changed_source.snapshot_id,
            "node_content_digest": changed_source.nodes[node_index].node_content_digest,
        }
    )
    assert (
        vf.project_subject(changed_subject, changed_source, changed).reason
        == "generated_call_evidence_inconsistent"
    )
    # Invalid transport is rejected atomically before creating prompt nodes.
    for change in ({"emitted_call_index": 99}, {"token_span": (2, 5)}):
        forged = response.model_copy(deep=True)
        forged.tokens.generated_calls = tuple(
            call.model_copy(update=change) if call.attempt_index == 0 else call
            for call in forged.tokens.generated_calls
        )
        empty = vf.Trace(episode_id="episode", agent=trace.agent, task=trace.task)
        with pytest.raises(ValueError):
            prepare_turn(empty, [vf.UserMessage(content="request")]).commit(forged)
        assert empty.nodes == []
    unqualified = restored.model_copy(deep=True)
    unqualified.nodes[node_index].generated_call_producer = None
    unqualified.nodes[node_index].generated_calls = tuple(
        call.model_copy(update={"parser_revision": "unqualified:parser"})
        for call in unqualified.nodes[node_index].generated_calls
    )
    unqualified_source = capture_trace_source(unqualified)
    unqualified_subject = subject.model_copy(
        update={
            "snapshot_id": unqualified_source.snapshot_id,
            "node_content_digest": unqualified_source.nodes[
                node_index
            ].node_content_digest,
        }
    )
    assert (
        vf.project_subject(unqualified_subject, unqualified_source, unqualified).reason
        == "generated_call_parser_provenance_unqualified"
    )


def test_generated_joint_and_missing_call_spans_remain_unsupported():
    from renderers.base import ParsedToolCall

    import verifiers.v1 as vf
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.clients.train import response_from_generate
    from verifiers.v1.graph import prepare_turn

    response = response_from_generate(
        {
            "completion_ids": [1, 2, 3, 4],
            "tool_calls": [
                ParsedToolCall(raw="a()", name="a", token_span=(0, 3)),
                ParsedToolCall(raw="b()", name="b", token_span=(0, 3)),
                ParsedToolCall(raw="c()", name="c"),
            ],
        },
        "model",
    )
    assert [call.span_fidelity for call in response.tokens.generated_calls] == [
        "joint",
        "joint",
        "unavailable",
    ]
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="calls")),
    )
    index = prepare_turn(trace, []).commit(response)
    source = capture_trace_source(trace)
    for ordinal in range(3):
        subject = vf.SubjectRef(
            kind="call",
            snapshot_id=source.snapshot_id,
            episode_id="episode",
            trace_id=trace.id,
            node_index=index,
            node_content_digest=source.nodes[index].node_content_digest,
            call_index=ordinal,
        )
        projection = vf.project_subject(subject, source, trace)
        assert projection.status == "unsupported" and not projection.intervals


def test_qwen35_parser_call_spans_follow_reasoning_and_exclude_stop_suffix():
    from renderers.parsing import parse_qwen35

    from verifiers.v1.clients.train import response_from_generate

    class Tokenizer:
        @staticmethod
        def decode(tokens, *, skip_special_tokens=False):
            return bytes(token for token in tokens if token < 256).decode()

    good = list(b"<function=send><parameter=id>1</parameter></function>")
    broken = list(b"malformed")
    original = [
        900,
        *b"reasoning",
        901,
        902,
        *good,
        903,
        902,
        *broken,
        903,
        902,
        *good,
        903,
        904,
        42,
    ]
    parsed = parse_qwen35(
        Tokenizer(),
        original,
        stop_ids={904},
        think_id=900,
        think_end_id=901,
        tool_call_id=902,
        tool_call_end_id=903,
    )
    assert len(parsed.tool_calls) == 3
    assert parsed.tool_calls[0].token_span[0] > original.index(901)
    assert parsed.tool_calls[-1].token_span[1] == original.index(904)
    assert [original[slice(*call.token_span)] for call in parsed.tool_calls] == [
        [902, *good, 903],
        [902, *broken, 903],
        [902, *good, 903],
    ]
    response = response_from_generate(
        {"completion_ids": original, "tool_calls": parsed.tool_calls}, "model"
    )
    assert len(response.tokens.generated_calls) == 3
    assert response.tokens.generated_calls[1].emitted_call_index is None
    assert all(
        call.span_fidelity == "exact" for call in response.tokens.generated_calls
    )


class LFMTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|tool_call_start|>": 100, "<|tool_call_end|>": 101}.get(token, -1)

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        if token_ids == [1]:
            return "[salesforce_note_create(parent_id='001001', title='Q1')]"
        return "content"


def test_train_client_config_serializes_selected_chat_template():
    config = TrainClientConfig(
        base_url="http://policy.invalid/v1",
        renderer_model_name="base-model",
        chat_template="selected {{ messages }}",
    )

    restored = TrainClientConfig.model_validate_json(config.model_dump_json())

    assert restored.chat_template == "selected {{ messages }}"


def test_lfm2_parser_recovers_pythonic_tool_call_without_executing_code():
    content, calls = LFM2ToolParser(LFMTokenizer()).extract([9, 100, 1, 101])

    assert content == [9]
    assert len(calls) == 1
    assert calls[0].name == "salesforce_note_create"
    assert calls[0].arguments == {"parent_id": "001001", "title": "Q1"}
    assert calls[0].status.value == "ok"
    assert calls[0].token_span == (1, 4)


def test_k2_ifm_parser_recovers_xml_tool_call_from_sampled_tokens():
    class Tokenizer:
        @staticmethod
        def encode(text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode())

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return bytes(token_ids).decode()

    raw = (
        "reasoning</ifm|think><ifm|tool_calls><ifm|tool_call>asana_get_task\n"
        "<ifm|arg_key>gid</ifm|arg_key><ifm|arg_value>asana_123</ifm|arg_value>"
        "<ifm|arg_key>limit</ifm|arg_key><ifm|arg_value>2</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    content, calls = K2IFMToolParser(Tokenizer()).extract(list(raw.encode()))

    assert bytes(content).decode() == "reasoning</ifm|think>"
    assert len(calls) == 1
    assert calls[0].name == "asana_get_task"
    assert calls[0].arguments == {"gid": "asana_123", "limit": 2}
    assert calls[0].status.value == "ok"


def test_k2_ifm_reasoning_parser_preserves_generated_thinking_field():
    reasoning, content = K2IFMReasoningParser(None).extract(
        "Inspect the task.</ifm|think><ifm|tool_calls>action</ifm|tool_calls>"
    )

    assert reasoning == "Inspect the task."
    assert content == "<ifm|tool_calls>action</ifm|tool_calls>"


def test_lfm2_bridge_restores_template_close_after_stripped_stop_token():
    class Tokenizer:
        bos_token_id = 5
        eos_token_id = 7

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert text == "\n"
            assert add_special_tokens is False
            return [8]

    class Renderer:
        _tokenizer = Tokenizer()

        @staticmethod
        def render(messages, *, tools, add_generation_prompt):
            assert messages == [{"role": "tool", "content": "created"}]
            assert tools is None
            assert add_generation_prompt is True
            from renderers import RenderedTokens

            return RenderedTokens(
                token_ids=[5, 9, 10],
                message_indices=[-1, 0, -1],
                message_roles=["tool"],
                message_tool_names=[None],
            )

    bridged = bridge_lfm2_tool_cycle(
        Renderer(),
        [1, 2],
        [3, 4],
        [{"role": "tool", "content": "created"}],
    )

    assert bridged is not None
    assert bridged.token_ids == [1, 2, 3, 4, 7, 8, 9, 10]
    assert bridged.message_indices == [-1, -1, -1, -1, -1, -1, 0, -1]


@pytest.mark.asyncio
async def test_renderer_pool_applies_template_and_fences_cache_identity(monkeypatch):
    import renderers
    from renderers import base

    ElasticRendererPool._renderers.clear()
    ElasticRendererPool._locks.clear()
    loaded = []
    created = []

    def load_tokenizer(model):
        tokenizer = SimpleNamespace(model=model, chat_template="artifact template")
        loaded.append(tokenizer)
        return tokenizer

    def create_renderer(tokenizer, config, *, chat_template_kwargs):
        created.append((tokenizer, config, chat_template_kwargs))
        return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(base, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(renderers, "create_renderer", create_renderer)
    selected = ElasticRendererPool(
        "base-model",
        DefaultRendererConfig(),
        chat_template="selected template",
        multiplex=1,
    )
    artifact = ElasticRendererPool(
        "base-model",
        DefaultRendererConfig(),
        chat_template=None,
        multiplex=1,
    )

    selected_slot = await selected.grow()
    artifact_slot = await artifact.grow()

    assert selected.key != artifact.key
    assert selected_slot.renderer.tokenizer.chat_template == "selected template"
    assert artifact_slot.renderer.tokenizer.chat_template == "artifact template"
    assert len(loaded) == 2
    assert len(created) == 2
