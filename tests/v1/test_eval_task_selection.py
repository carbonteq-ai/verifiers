import pytest
from pydantic import ValidationError

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import EvalRunInfo


def test_eval_config_accepts_exact_task_keys() -> None:
    config = EvalConfig(task_keys=["task-b", "task-a"], num_tasks=2, push=False)
    assert config.task_keys == ["task-b", "task-a"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"task_keys": []}, "task_keys cannot be empty"),
        ({"task_keys": ["a", "a"]}, "task_keys must be unique"),
        ({"task_keys": ["a"], "shuffle": True}, "cannot be combined with shuffle"),
        ({"task_keys": ["a"], "num_tasks": 2}, "equal len"),
    ],
)
def test_eval_config_rejects_ambiguous_exact_selection(values, message) -> None:
    with pytest.raises(ValidationError, match=message):
        EvalConfig(**values, push=False)


def test_evaluation_run_info_retains_repetition_identity() -> None:
    info = EvalRunInfo(id="eval-1", repetition_index=2)

    assert info.model_dump(mode="json") == {
        "type": "eval",
        "id": "eval-1",
        "name": None,
        "repetition_index": 2,
    }
