from types import SimpleNamespace

import pytest
from renderers import DefaultRendererConfig

from verifiers.v1.clients.train import ElasticRendererPool
from verifiers.v1.configs.client import TrainClientConfig


def test_train_client_config_serializes_selected_chat_template():
    config = TrainClientConfig(
        base_url="http://policy.invalid/v1",
        renderer_model_name="base-model",
        chat_template="selected {{ messages }}",
    )

    restored = TrainClientConfig.model_validate_json(config.model_dump_json())

    assert restored.chat_template == "selected {{ messages }}"


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
