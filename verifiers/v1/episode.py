"""The episode — one run's traces plus their shared standing, whole."""

import uuid
from typing import Annotated, Any, Generic, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from verifiers.v1.assessments import AssessmentBatch, SourceSnapshot
from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.credit import CreditAssignment
from verifiers.v1.graph import RECORD_FLOAT_DECIMALS
from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.trace import EXCLUDE_FIELDS, AgentConfigT, Error, Trace, TraceTask
from verifiers.v1.types import Usage


class EnvInfo(BaseModel):
    """The env that ran the episode, self-describing without the run's config."""

    id: str = ""
    """`EnvConfig.env_id`, e.g. `agentic-judge+gsm8k`."""
    name: str | None = None
    """The name the consumer runs the env under (e.g. an orchestrator env key)."""


class GroupInfo(BaseModel):
    """The consumer-defined rollout group this episode belongs to."""

    id: str


class PolicySpan(BaseModel):
    """Live policy versions spanned while generating an episode."""

    start: int
    end: int

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("policy span end must be at least its start")
        return self

    @property
    def drift(self) -> int:
        return self.end - self.start


class TrainWorkInfo(BaseModel):
    type: Literal["train"] = "train"
    step: int
    """The training step at which the episode was dispatched."""
    policy: PolicySpan | None = None
    """The live policy used for generation; None for frozen-policy work."""


class EvalWorkInfo(BaseModel):
    type: Literal["eval"] = "eval"
    step: int
    """The training step at which the episode was dispatched."""
    policy: PolicySpan | None = None
    """The live policy used for generation; None for frozen-policy work."""


WorkInfo = Annotated[TrainWorkInfo | EvalWorkInfo, Field(discriminator="type")]


class TrainRunInfo(BaseModel):
    """A training run and the kind of work this episode performed for it."""

    type: Literal["train"] = "train"
    id: str
    name: str | None = None
    work: WorkInfo


class EvalRunInfo(BaseModel):
    """A standalone evaluation run."""

    type: Literal["eval"] = "eval"
    id: str
    name: str | None = None
    repetition_index: int | None = Field(default=None, ge=0)
    """Stable slot of this task repetition within the standalone evaluation."""


RunInfo = Annotated[TrainRunInfo | EvalRunInfo, Field(discriminator="type")]


class Episode(BaseModel, Generic[DataT, StateT, AgentConfigT]):
    """The artifact Env.run produces. Contains multiple agents' traces."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    env: EnvInfo = Field(default_factory=EnvInfo)
    """The env that produced this episode."""
    task: TraceTask[DataT]
    """The task dispatched to the env, including when no trace was produced."""
    group: GroupInfo | None = None
    """Consumer-assigned rollout group, if this episode was grouped."""
    run: RunInfo | None = None
    """The run this episode belongs to, consumer-stamped."""
    ok: bool = False
    """Whether the episode completed successfully."""
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest to newest."""
    traces: list[Trace[DataT, StateT, AgentConfigT]] = Field(default_factory=list)
    """Every agent's trace, in completion order."""
    assessment_batches: list[AssessmentBatch] = Field(default_factory=list)
    """Episode-wide assessment attempts over one or more native traces."""
    credit_assignments: list[CreditAssignment] = Field(default_factory=list)
    """Episode-wide domain assignments retaining their accepted source evidence."""
    credit_errors: list[str] = Field(default_factory=list)
    """Episode assignment planning errors, separate from solver errors."""
    assessment_errors: list[str] = Field(default_factory=list)
    """Assessment infrastructure/availability reason codes, separate from execution."""
    assessment_finalization_state: Literal["completed", "failed", "not_run"] | None = (
        None
    )
    """Finalization evidence availability when episode assessment was planned."""

    @model_validator(mode="before")
    @classmethod
    def restore_assessment_archive(cls, value):
        from verifiers.v1.assessment_archive import restore_history

        return restore_history(value) if isinstance(value, dict) else value

    @model_serializer(mode="wrap")
    def serialize_assessment_archive(self, handler, info):
        from verifiers.v1.assessment_archive import normalize_history

        data = handler(self)
        return normalize_history(data) if info.mode == "json" else data

    @field_validator("assessment_batches")
    @classmethod
    def retained_assessment_sources(
        cls, batches: list[AssessmentBatch]
    ) -> list[AssessmentBatch]:
        for batch in batches:
            if not isinstance(batch.source, SourceSnapshot):
                raise ValueError(  # noqa: TRY004 - Pydantic schema validation
                    "native assessment persistence requires the full retained source"
                )
        return batches

    @field_validator("credit_assignments")
    @classmethod
    def retained_assignment_sources(
        cls, assignments: list[CreditAssignment]
    ) -> list[CreditAssignment]:
        for assignment in assignments:
            if not isinstance(assignment.request.source, SourceSnapshot):
                raise ValueError(  # noqa: TRY004 - Pydantic schema validation
                    "native assignment persistence requires the full retained source"
                )
        return assignments

    def record_run(self, run: RunInfo) -> None:
        """Record the run identity on the dispatched episode."""
        self.run = run

    @property
    def last_error(self) -> Error | None:
        """The last episode-level error captured across attempts."""
        return self.errors[-1] if self.errors else None

    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed across every trace's model calls;
        judge/off-graph usage stays on the traces (`Trace.extra_usage`)."""
        return Usage.aggregate(u for t in self.traces if (u := t.usage) is not None)

    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), summed across traces."""
        return sum(t.num_input_tokens for t in self.traces)

    @property
    def num_output_tokens(self) -> int:
        """Model-generated tokens across all turns, summed across traces."""
        return sum(t.num_output_tokens for t in self.traces)

    @property
    def num_total_tokens(self) -> int:
        """Final sequence lengths per branch, summed across traces."""
        return sum(t.num_total_tokens for t in self.traces)

    @property
    def num_turns(self) -> int:
        """Sampled turns, summed across traces."""
        return sum(t.num_turns for t in self.traces)

    @property
    def by_agent(self) -> dict[str, list[Trace[DataT, StateT, AgentConfigT]]]:
        """Traces grouped by agent name (e.g. n solvers), in completion order."""
        grouped: dict[str, list[Trace[DataT, StateT, AgentConfigT]]] = {}
        for trace in self.traces:
            grouped.setdefault(trace.agent.name, []).append(trace)
        return grouped

    def to_record(
        self, float_decimals: int | None = RECORD_FLOAT_DECIMALS
    ) -> dict[str, Any]:
        """JSON record without raw trace tensors, which remain on the msgpack wire.
        Per-token float streams are rounded to `float_decimals` (`None` keeps every digit)."""
        return self.model_dump(
            mode="json",
            exclude={"traces": {"__all__": EXCLUDE_FIELDS}},
            exclude_none=True,
            context={"float_decimals": float_decimals},
        )


WireEpisode = Episode[WireTaskData, State, WireAgentConfig]
"""Record loader for consumers without the run's packages: unknown task fields
survive in `task.model_extra`, agent configs parse loose (`WireAgentConfig`)."""
