"""
Learning-rate schedules for phases that occur more than once in a training schedule.

A ProtoPNet schedule repeats its joint phase after every projection, and all joint
phases share one optimizer. A regular torch scheduler therefore keeps decaying from one
joint phase to the next: with the default StepLR (step size 5, gamma 0.1) and 5-epoch
joint phases, every joint phase starts at a tenth of the previous one's learning rate.
`PhaseRestartingLR` instead restarts its schedule at the start of every occurrence of
its phase.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch

# Maps (epoch within the phase, number of epochs in the phase) to a multiplier on the
# base learning rate.
LRFactor = Callable[[int, int], float]

JOINT_LR_SCHEDULES = ("step", "step-reset", "cosine")


@dataclass(frozen=True)
class StepDecay:
    """Multiplies the learning rate by `gamma` every `step_size` epochs of the phase."""

    step_size: int
    gamma: float

    def __call__(self, epoch_in_phase: int, phase_length: int) -> float:
        return self.gamma ** (epoch_in_phase // self.step_size)


@dataclass(frozen=True)
class CosineDecay:
    """Anneals the learning rate from its base value to `min_factor` times it over the phase."""

    min_factor: float = 0.0

    def __call__(self, epoch_in_phase: int, phase_length: int) -> float:
        progress = epoch_in_phase / phase_length
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_factor + (1 - self.min_factor) * cosine


class PhaseRestartingLR:
    """
    A learning-rate schedule that restarts every time its phase starts.

    `ClassificationBackpropPhase` calls `start_phase` before each occurrence of its phase
    and `step` after each epoch, as it does for a torch scheduler. Each parameter group's
    learning rate is its base rate times `lr_factor(epoch_in_phase, phase_length)`.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_factor: LRFactor):
        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._phase_length: Optional[int] = None
        self._epoch_in_phase = 0

    def start_phase(self, phase_length: int) -> None:
        """Return to the base learning rates at the start of an occurrence of the phase."""
        if phase_length < 1:
            raise ValueError(
                f"A phase must last at least one epoch, not {phase_length}."
            )
        self._phase_length = phase_length
        self._epoch_in_phase = 0
        self._apply_lr_factor()

    def step(self) -> None:
        """Move on to the next epoch of the phase."""
        if self._phase_length is None:
            raise RuntimeError("start_phase() must be called before step().")
        self._epoch_in_phase += 1
        self._apply_lr_factor()

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _apply_lr_factor(self) -> None:
        factor = self.lr_factor(self._epoch_in_phase, self._phase_length)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


def joint_lr_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    step_size: int,
    gamma: float,
) -> Union[torch.optim.lr_scheduler.StepLR, PhaseRestartingLR]:
    """
    Creates the learning-rate scheduler that all joint phases share.

    Args:
        name: One of JOINT_LR_SCHEDULES.
            - "step": StepLR(step_size, gamma), never restarted, so the learning rate
              keeps decaying from one joint phase to the next. This matches the
              original ProtoPNet and is the default.
            - "step-reset": the same step decay, restarted in every joint phase. When
              `step_size` equals the length of a joint phase (as with the defaults of
              train-protopnet and train-tesnet), the learning rate stays at its base
              value.
            - "cosine": cosine annealing from the base learning rate towards zero within
              every joint phase. `step_size` and `gamma` are not used.
        optimizer: The joint phases' optimizer.
        step_size: Epochs between learning-rate decays for "step" and "step-reset".
        gamma: Decay factor for "step" and "step-reset".
    """
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "step-reset":
        return PhaseRestartingLR(optimizer, StepDecay(step_size=step_size, gamma=gamma))
    if name == "cosine":
        return PhaseRestartingLR(optimizer, CosineDecay())
    raise ValueError(
        f"Unknown joint learning-rate schedule {name!r}; "
        f"expected one of {', '.join(JOINT_LR_SCHEDULES)}."
    )
