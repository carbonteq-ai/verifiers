from types import SimpleNamespace

import pytest
from renderers import DefaultRendererConfig

from verifiers.v1.clients.renderer_extensions import (
    LFM2ToolParser,
    bridge_lfm2_tool_cycle,
)
from verifiers.v1.clients.train import ElasticRendererPool
from verifiers.v1.configs.client import TrainClientConfig


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
