from typing import ClassVar

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.episode import WireEpisode
from verifiers.v1.types import SamplingConfig


class BaseRequest(BaseModel):
    """`method` is sent as its own route frame, not as payload data."""

    method: ClassVar[str]


class BaseResponse(BaseModel):
    success: bool = True
    error: str | None = None


class HealthRequest(BaseRequest):
    method: ClassVar[str] = "health"


class HealthResponse(BaseResponse):
    pass


class CancelRequest(BaseRequest):
    """Abort an in-flight run by its wire ``request_id``. Idempotent — a
    finished or unknown run cancels successfully with ``cancelled=False``."""

    method: ClassVar[str] = "cancel"
    request_id: str


class CancelResponse(BaseResponse):
    cancelled: bool = False
    """Whether a live run was found and aborted."""


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself.

    ``task_data`` and ``task_config`` are the dumped values the server validates
    into the taskset's declared types.  A taskset may derive per-task config while
    loading (for example, a task-specific tool allowlist), so data alone is not a
    lossless task representation.  ``task_config`` remains optional for older
    clients whose tasks all use the environment's static config.
    """

    method: ClassVar[str] = "run"
    task_data: dict
    task_config: dict | None = None
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    episode: SerializeAsAny[WireEpisode] | None = None
    """The rollout's episode — its standing (`id`/`env`/`errors`, carrying
    episode-level errors even when no trace minted) inlined next to its flat,
    self-contained traces; task-specific data preserved in `model_extra`."""
