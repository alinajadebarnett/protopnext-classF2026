"""
Read a ProtoPNet training schedule from a YAML file.

A schedule config file has three optional sections::

    stages:              # settings for each backprop stage
      warm:
        trainable: [add_on_layers, prototype_layer]
        lr: {add_on_layers: 0.003, prototype_layer: 0.003}
      joint:
        trainable: [backbone, add_on_layers, prototype_layer]
      last_only:
        trainable: [prototype_prediction_head]

    freeze_batchnorm: false

    schedule:            # the order in which stages run
      - {stage: warm, epochs: 5}
      - {stage: joint, epochs: 5}
      - {stage: project}
      - repeat: 20
        stages:
          - {stage: last_only, epochs: 10}
          - {stage: joint, epochs: 5}
          - {stage: project}
      - {stage: last_only, epochs: 10}

Anything left out keeps the ProtoPNet default. ``load_schedule_config`` returns
keyword arguments for ``ProtoPNetTrainingSchedule``; detailed checks (unknown
component names, learning rates for frozen components, invalid stage orders)
happen there, so a config file and the Python API give the same errors.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

#: Top-level sections allowed in a schedule config file.
CONFIG_SECTIONS = ("stages", "freeze_batchnorm", "schedule")

#: Settings allowed for each stage under ``stages``.
STAGE_SETTINGS = ("trainable", "lr")


def load_schedule_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a schedule config YAML file.

    Returns a dict that may contain ``trainable_components``,
    ``stage_learning_rates``, ``freeze_batchnorm`` and ``stage_order``, ready to
    pass to ``ProtoPNetTrainingSchedule``.
    """
    path = Path(path)
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{path} is not valid YAML: {e}") from e
    return schedule_kwargs_from_config(config, source=str(path))


def schedule_kwargs_from_config(
    config: Optional[Mapping[str, Any]], source: str = "schedule config"
) -> Dict[str, Any]:
    """Turn an already-parsed config (e.g. from YAML) into schedule keyword arguments."""
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError(
            f"{source}: the top level must be a mapping with the sections "
            f"{list(CONFIG_SECTIONS)}, got {type(config).__name__}"
        )

    unknown = set(config) - set(CONFIG_SECTIONS)
    if unknown:
        raise ValueError(
            f"{source}: unknown section(s) {sorted(unknown)}. "
            f"Valid sections: {list(CONFIG_SECTIONS)}"
        )

    kwargs: Dict[str, Any] = {}

    stages = config.get("stages")
    if stages is not None:
        if not isinstance(stages, Mapping):
            raise ValueError(
                f"{source}: 'stages' must map stage names to their settings, "
                f"got {type(stages).__name__}"
            )
        trainable: Dict[str, Any] = {}
        learning_rates: Dict[str, Any] = {}
        for stage_name, settings in stages.items():
            if settings is None:
                continue
            if not isinstance(settings, Mapping):
                raise ValueError(
                    f"{source}: settings for stage {stage_name!r} must be a mapping "
                    f"with keys {list(STAGE_SETTINGS)}"
                )
            unknown = set(settings) - set(STAGE_SETTINGS)
            if unknown:
                raise ValueError(
                    f"{source}: unknown setting(s) {sorted(unknown)} for stage "
                    f"{stage_name!r}. Valid settings: {list(STAGE_SETTINGS)}"
                )
            if "trainable" in settings:
                trainable[stage_name] = settings["trainable"]
            if "lr" in settings:
                lr = settings["lr"]
                if not isinstance(lr, (Mapping, str)):
                    raise ValueError(
                        f"{source}: 'lr' for stage {stage_name!r} must map "
                        "components to learning rates"
                    )
                learning_rates[stage_name] = lr
        if trainable:
            kwargs["trainable_components"] = trainable
        if learning_rates:
            kwargs["stage_learning_rates"] = learning_rates

    if "freeze_batchnorm" in config:
        freeze_batchnorm = config["freeze_batchnorm"]
        if not isinstance(freeze_batchnorm, bool):
            raise ValueError(
                f"{source}: 'freeze_batchnorm' must be true or false, "
                f"got {freeze_batchnorm!r}"
            )
        kwargs["freeze_batchnorm"] = freeze_batchnorm

    if "schedule" in config:
        schedule = config["schedule"]
        if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
            raise ValueError(
                f"{source}: 'schedule' must be a list of stages, "
                f"got {type(schedule).__name__}"
            )
        kwargs["stage_order"] = list(schedule)

    return kwargs


def scale_stage_order(
    stage_order: Sequence[Any], *, multiplier: int = 1, verify: bool = False
) -> List[Any]:
    """
    Copy of a stage order with every ``epochs`` value multiplied by ``multiplier``.

    With ``verify=True`` every stage with epochs runs for 1 epoch and every repeat
    block runs once, so all stages can be checked quickly. Items that are not
    well formed are copied unchanged, so validation reports them later.
    """
    scaled = copy.deepcopy(list(stage_order))

    def scale_stage(item: Any) -> None:
        if not isinstance(item, dict):
            return
        epochs = item.get("epochs")
        if isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0:
            item["epochs"] = 1 if verify else epochs * multiplier

    for item in scaled:
        if isinstance(item, dict) and "repeat" in item:
            repeat = item["repeat"]
            if (
                verify
                and isinstance(repeat, int)
                and not isinstance(repeat, bool)
                and repeat > 0
            ):
                item["repeat"] = 1
            inner = item.get("stages")
            if isinstance(inner, list):
                for inner_item in inner:
                    scale_stage(inner_item)
        else:
            scale_stage(item)
    return scaled
