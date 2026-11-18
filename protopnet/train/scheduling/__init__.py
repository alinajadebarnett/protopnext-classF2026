from typing import List

from .scheduling import (  # noqa F401
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    PrunePrototypesPhase,
    RSampleInitPhase,
    TrainLayersUsingProtoPNetNames,
)
from .types import (  # noqa F401
    EarlyStopping,
    IterativePhase,
    Phase,
    PhaseType,
    PostPhaseSummary,
    PrePhaseSummary,
    SetTrainingLayers,
    StepContext,
    TrainingSchedule,
)


def __getattr__(name: str):
    # Loaded on demand so building a model does not import wandb.
    if name == "ProjectPatienceEarlyStopping":
        from .early_stopping import ProjectPatienceEarlyStopping

        return ProjectPatienceEarlyStopping
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def repeated(phases: List[Phase], iterations: int) -> IterativePhase:
    return IterativePhase(phases=phases, iterations=iterations)
