from types import SimpleNamespace

from pydantic import ConfigDict

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.task import Task, TaskData


class DerivedData(TaskData):
    marker: str


class DerivedConfig(TaskConfig):
    model_config = ConfigDict(frozen=True)

    allowed_tools: tuple[str, ...] = ()


class DerivedTask(Task[DerivedData, dict, DerivedConfig]):
    pass


def _server() -> EnvServer:
    server = EnvServer.__new__(EnvServer)
    server.data_cls = DerivedData
    server.task_cls = DerivedTask
    server.env = SimpleNamespace(
        config=SimpleNamespace(
            taskset=SimpleNamespace(task=DerivedConfig(allowed_tools=("catalog-default",)))
        )
    )
    return server


def test_env_server_preserves_per_task_derived_config() -> None:
    task = _server()._build_task(
        {"marker": "row-1"},
        {"allowed_tools": ["row-specific-tool"]},
    )

    assert task.data.marker == "row-1"
    assert task.config.allowed_tools == ("row-specific-tool",)


def test_env_server_accepts_legacy_data_only_request() -> None:
    task = _server()._build_task({"marker": "row-1"})

    assert task.config.allowed_tools == ("catalog-default",)
