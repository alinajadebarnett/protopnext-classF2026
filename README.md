# ProtoPNext

**This is a prerelease codebase. All APIs, interfaces, and implementations are subject to change without notice.**

ProtoPNext is a research library for prototypical part networks (PPNs) that provides implementations of various prototype-based interpretable models including ProtoPNet, TesNet, Deformable ProtoPNet, ProtoTree, and ST-ProtoPNet.

## Installing

`protopnext` currently supports python 3.8.
We recommend creating a dedicated python environment for development (i.e., with `virtualenv` or `conda`).

### Install Dependencies

You need to choose between CPU and CUDA-based development.
If you have an NVidia GPU, you should install the CUDA torch dependencies so you can switch between CPU and GPU execution.
Otherwise, use CPU.

For CPU:

```{sh}
pip install -r env/requirements-frozen.txt --extra-index-url=https://download.pytorch.org/whl/cpu
```

For CUDA:

```{sh}
pip install -r env/requirements-frozen.txt --extra-index-url=https://download.pytorch.org/whl/cu117
```

## Training Your First ProtoPNet

This will walk you through the commands to setup a CUB-200 dataset and train ProtoPNet.

### CUB-200 Dataset Prep

1. Download the dataset CUB_200_2011.tgz from https://www.vision.caltech.edu/datasets/cub_200_2011/ using the command `wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz`
2. Unpack CUB_200_2011.tgz. 
Check to see if the file unpacked properly
3. Run `python -m protopnet datasets create-splits /path/to/unpacked/CUB_200_2011`. This will create `train`, `validation`, and `test` directories in your `CUB_200_2011` directory that match the splits used for ProtoPNeXt.

### Training Your first ProtoPNet

On the isolated environment you set up with CUDA available, run the following:

1. Log in to the weights and biases with `wandb login`
2. Set your CUB_200 directory to the variable `export CUB200_DIR=/path/to/unpacked/CUB_200_2011`
3. Run `python -m protopnet train-protopnet`. See `protopnet.train_protopnet.main` for the default parameters. (Suggestion: It might be worth to change the `phase_multiplier` parameter value to something small like 1-5 for your first run of ProtoPNet to reduce the number of epochs in each training phase, but this is not necessary).

Local logging shows the progress of the training in Epochs.
In Weights and Biases, you can view the training curves.

### Visualizing Model Reasoning

There are several options for visualizing model reasoning. Start by training a model, for example:

`protopnet train-protopnet --dataset=shapes_2d --proto-channels=32 --num-prototypes-per-class=5 --output-dir=analysis`

Once trained, use the `viz` argument to render the prototypes:

`protopnet viz render-prototypes --model-path="/get/from/training/logs" --dataset=shapes_2d --output-dir=analyses`

This will produce multiple visualization for each prototype including:

- Original prototype image
- Heatmap showing activation
- Overlaid heatmap on the original image
- Original image with bounding box around the high-activation region
- In cases where parts are specified in the dataset, vizualization of part locations.

#### Local Analysis

It is also possible to do `local` analyses, focusing on a specific test instances and associated classifications.

`protopnet viz local --model-path="/get/from/training/logs" --dataset=shapes_2d --sample=<num_samples> --output-dir=analyses`

This will create a set of vizualizations including:

- Test image alongside activation prototypes
- Top-k prototypes that influenced the classification
- Similarity scores and connections to classes
- Contribution of each prototype to the final classification.

The `--sample` parameter is used to limit the number of test images that will be analyzed.

#### Global Analysis

Finally, `global` analyses are possible that examine prototypes behavior across the entire dataset. This will find the top N similar samples to the generated prototypes. It also provides heatmaps and overlays.

`protopnet viz global --model-path="/get/from/training/logs" --dataset=shapes_2d --sample=<num_prototypes> --output-dir=analysis`

If you call these without first having rendered prototypes, they will be automatically rendered for you.

## Code Organization

```
protopnet/
├── models/              # Model implementations
│   ├── vanilla_protopnet.py
│   ├── deformable_protopnet.py
│   └── ...
├── datasets/            # Dataset loaders and utilities
│   ├── torch_extensions.py
│   ├── cars_cropped.py
│   └── ...
├── train/              # Auto-training infrastructure
│   ├── scheduling/     # Training phase scheduling
│   ├── logging/        # Metrics and W&B integration
│   └── checkpointing.py
├── utilities/          # General utilities
│   ├── trainer_utilities.py
│   ├── visualization_utilities.py
│   └── ...
├── pretrained/         # Pretrained backbone implementations
├── activations.py      # Prototype activation functions
├── backbones.py        # Backbone model construction
├── embedding.py        # Feature embedding layers
├── prediction_heads.py # Classification heads
├── prototype_layers.py # Prototype computation layers
└── cli.py              # Command-line interface
```

### Core Components vs Training Infrastructure

The codebase is organized into two main categories:

**Core Library Components** are used for constructing models:

- `activations.py` - Cosine, L2, and other prototype activation functions
- `backbones.py` - Backbone model construction and configuration
- `embedding.py` - Feature embedding and addon layers  
- `prediction_heads.py` - Classification heads for prototype models
- `prototype_layers.py` - Core prototype computation and projection
- `models/` - Complete model implementations
- `datasets/` - Dataset loading and preprocessing
- `pretrained/` - Pretrained model implementations
- `utilities/` - General utilities for visualization, preprocessing, etc.

**Auto-Training Infrastructure** (`train/` package) is used for creating training loops:

- `scheduling/` - Multi-phase training schedules and early stopping
- `logging/` - Metrics tracking, W&B integration, and logging
- `checkpointing.py` - Model saving and loading utilities
- `types.py` - Training-related protocols and interfaces

## Key Concepts

### Prototype Models

All models inherit from `ProtoPNet` base class and consist of:

- **Backbone**: Feature extraction (ResNet, DenseNet, VGG, etc.)
- **Add-on layers**: Feature processing before prototype computation
- **Prototype layer**: Computes similarity between features and learned prototypes
- **Prediction head**: Maps prototype activations to class predictions

### Training Phases

Training alternates between multiple phases:

- **Warm-up**: Train only prototype layers (backbone frozen)
- **Joint**: End-to-end training of all components
- **Project**: Update prototypes to match training examples
- **Last-only**: Train only the classification head

Which components train in each phase, how many epochs each phase runs, and the learning rate of each component can be changed per phase; see [Configuring the Training Schedule](#configuring-the-training-schedule).

### Early Stopping

Custom early stopping for prototype models:

- **Project patience**: Stop after N projections without improvement

## Command Line Interface

The library provides a CLI to train models based on command line configuration, which can be used as a starting point for experimenting with ProtoPNet models.
Custom models will require their own configurations.

```bash
# Train vanilla ProtoPNet
python -m protopnet train-protopnet --dataset cub200 --backbone resnet50

# Train deformable ProtoPNet  
python -m protopnet train-deformable --dataset cub200 --activation-function cosine

# Evaluate models
python -m protopnet eval models.csv --dataset cub200 --output results.csv

# Generate visualizations
python -m protopnet visualization model.pth --dataset cub200
```

### Configuring the Training Schedule

`train-protopnet` trains in stages: **warm** → **joint** → **project** → a repeated cycle of (**last_only** → **joint** → **project**) → a final **last_only** stage.
The options below change one stage at a time.
Every option is optional: if you leave it out, that stage keeps the original ProtoPNet setting, so existing commands behave exactly as before.

Component names used by these options:

| Name | Component |
|---|---|
| `backbone` | feature extractor (e.g. ResNet) |
| `add_on_layers` | add-on layers between backbone and prototypes |
| `prototype_layer` | the prototype vectors |
| `prototype_prediction_head` | final classification layer |

#### Which components train in each stage

| Option | Default |
|---|---|
| `--warm-trainable` | `add_on_layers,prototype_layer` |
| `--joint-trainable` | `backbone,add_on_layers,prototype_layer` |
| `--last-only-trainable` | `prototype_prediction_head` |
| `--freeze-batchnorm` | off |

Give a comma-separated list (no spaces) of the components to train; all other components are frozen for that stage.
Each stage's optimizer contains exactly its trainable components.
`--freeze-batchnorm` also keeps the BatchNorm running statistics of frozen components fixed; `requires_grad=False` alone does not stop them from changing.

#### Epochs per stage

| Option | Default |
|---|---|
| `--warm-epochs` | 5 |
| `--preproject-joint-epochs` | 5 |
| `--joint-epochs-per-cycle` | 5 |
| `--last-only-epochs-per-cycle` | 10 |
| `--num-cycles` | 20 |
| `--final-last-only-epochs` | same as `--last-only-epochs-per-cycle` |

All epoch counts are multiplied by `--phase-multiplier`. A stage set to `0` is skipped. `--verify` ignores these options and runs one epoch per stage.

#### Learning rate per stage

| Option | Stage |
|---|---|
| `--warm-lr` | warm |
| `--joint-lr` | joint |
| `--last-only-lr` | last_only |

Give `component=rate` pairs separated by commas, e.g. `--joint-lr add_on_layers=0.001,backbone=0.00005`.
Components you do not list keep their normal rate (`backbone` 1e-4, `add_on_layers` 3e-3, `prototype_layer` 3e-3, `prototype_prediction_head` 1e-4, each scaled by `--lr-multiplier`).
Rates given here are used as written and are **not** scaled by `--lr-multiplier`.
A rate can only be set for a component that trains in that stage; the joint stage's step-decay scheduler starts from the rate you set.

#### Examples

```bash
# Also train the add-on layers in the last-only stages, with their own learning rate
python -m protopnet train-protopnet --last-only-trainable prototype_prediction_head,add_on_layers --last-only-lr add_on_layers=0.0005

# Skip warm-up, keep the backbone frozen throughout, and run a short schedule
python -m protopnet train-protopnet --warm-epochs 0 --joint-trainable add_on_layers,prototype_layer --num-cycles 2 --final-last-only-epochs 15

# Freeze BatchNorm statistics in frozen components
python -m protopnet train-protopnet --freeze-batchnorm
```

To check a configuration without training, add `--dry-run`: the resolved schedule and a `Learning rates per stage` line are printed to the log.

Invalid settings stop the run before training with a clear error, for example a misspelled component name (`--warm-trainable backbon`), a negative epoch count, or a learning rate for a component that is frozen in that stage (`--warm-lr backbone=0.1`).

#### Schedule config files

Instead of command-line options, the whole schedule can be written in a YAML file and passed with `--schedule-config`.
A config file can also change the **order** of stages, add or remove stages, and change what repeats, without editing code.

```bash
python -m protopnet train-protopnet --schedule-config configs/schedules/protopnet_default.yaml
```

`configs/schedules/protopnet_default.yaml` reproduces the default schedule exactly; copy it as a starting point.
`configs/schedules/example_frozen_backbone.yaml` shows a customised schedule.

```yaml
stages:                      # settings per stage (all optional)
  warm:
    trainable: [add_on_layers, prototype_layer]
  joint:
    trainable: [add_on_layers, prototype_layer]   # keep the backbone frozen
    lr: {add_on_layers: 0.001}
  last_only:
    trainable: [prototype_prediction_head, add_on_layers]
    lr: {add_on_layers: 0.0005}

freeze_batchnorm: true

schedule:                    # order in which stages run
  - {stage: warm, epochs: 3}
  - {stage: joint, epochs: 5}
  - {stage: project}
  - repeat: 5
    stages:
      - {stage: last_only, epochs: 5}
      - {stage: joint, epochs: 3}
      - {stage: project}
  - {stage: last_only, epochs: 15}
```

| Section | Meaning |
|---|---|
| `stages.<stage>.trainable` | components trained in `warm`, `joint` or `last_only` (same as `--*-trainable`) |
| `stages.<stage>.lr` | learning rate per component in that stage (same as `--*-lr`) |
| `freeze_batchnorm` | `true` / `false` (same as `--freeze-batchnorm`) |
| `schedule` | list of `{stage: <name>, epochs: <n>}` (`warm`, `joint`, `last_only`), `{stage: project}`, and `{repeat: <n>, stages: [...]}` blocks |

Rules:

- Every section is optional; anything left out keeps the default. Without `schedule`, the standard stage order and lengths are used.
- A stage with `epochs: 0` and a block with `repeat: 0` are skipped. `project` takes no epochs. Repeat blocks cannot be nested.
- All stages of the same name share one optimizer, as in the default schedule (e.g. every `joint` stage continues the same Adam state and step-decay scheduler).
- Epochs in the file are multiplied by `--phase-multiplier`. With `--verify`, every stage runs one epoch and every repeat block runs once.
- `--schedule-config` cannot be combined with the per-stage options above (`--*-epochs`, `--num-cycles`, `--*-trainable`, `--*-lr`, `--freeze-batchnorm`); the run stops with an error naming the conflicting option.
- Mistakes are reported before training with their position, e.g. `item 4, stage 2: unknown stage 'lastonly'`.

Add `--dry-run` to print the resulting schedule, the trainable components and the learning rates of every stage without training.

## Datasets

Available datasets:

- **cub200**: CUB-200-2011 birds dataset
- **cub200_cropped**: Cropped version using bounding boxes
- **cars_cropped**: Stanford Cars with cropping
- **dogs**: Stanford Dogs dataset

## Library

### Training Phase System

Training alternates between *phases*, which are a series of *steps*.
`ClassificationBackpropPhase` is the standard gradient descent training phase, but configureswhich network layers to train.
Other phases have special non-gradient behaviors, like projection.

For example, the `ProtoPNetTrainingSchedule` has the following layout:

- **Warm-up**: `ClassificationBackpropPhase` training only prototype layers (and, optionally, last layer)
- **Joint**: `ClassificationBackpropPhase` training all layers
- **Project**: `ProjectPhase` - one epoch updating the prototypes to match the closest sample patches
- **Last-only**: `ClassificationBackpropPhase` training only the classification head

The same options as the CLI are available as arguments of `ProtoPNetTrainingSchedule`:

```python
schedule = ProtoPNetTrainingSchedule(
    ...,
    num_warm_epochs=0,                          # 0 skips the stage
    final_last_only_epochs=30,                  # default: last_only_epochs_per_project
    trainable_components={"last_only": ["prototype_prediction_head", "add_on_layers"]},
    stage_learning_rates={"last_only": {"add_on_layers": 5e-4}},
    freeze_batchnorm=True,
)
schedule.stage_trainable_components  # resolved components per stage
schedule.stage_learning_rates        # resolved learning rates per stage
```

A custom stage order can be passed as `stage_order` (the same list as the `schedule` section of a config file), and a config file can be loaded with `protopnet.schedule_config.load_schedule_config`:

```python
from protopnet.schedule_config import load_schedule_config

schedule = ProtoPNetTrainingSchedule(
    ...,
    **load_schedule_config("configs/schedules/protopnet_default.yaml"),
)
```

Which layers a `ClassificationBackpropPhase` trains is set by its `set_training_layers_fn`:

- `TrainLayersByComponent(trainable=[...], freeze_batchnorm=False)` trains the listed top-level components and freezes the rest. Unknown component names raise an error.
- `TrainOptimizerLayers()` (the default) trains exactly the parameters in the phase's optimizer, across all of its parameter groups.

## Contributors

- Frank Willard
- Maximilian Machado
- Emanuel Mokel
- Adam Costarino
- Jon Donnelly
- Zhicheng Guo
- Dennis Tang
- Julia Yang
- Giyoung Kim
- Alina Jade Barnett

## References

1. Chen et al. This Looks Like That: Deep Learning for Interpretable Image Recognition. NeurIPS, 2019.
2. Donnelly et al. Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes. ICCV, 2022.
3. Nauta et al. Neural prototype trees for interpretable fine-grained image recognition. ICCV, 2021.
4. Wang et al. Interpretable Image Recognition by Constructing Transparent Embedding Space. ICCV, 2021.
5. Wang et al. Learning support and trivial prototypes for interpretable image classification. ICCV, 2023
6. Huang et al. Evaluation and improvement of interpretability for self-explainable part-prototype networks. ICCV, 2023.
7. Willard et al. This looks better than that: Better interpretable models with protopnext. arXiv preprint arXiv:2406.14675 (2024).
8. Moffett et al. Cosine Similarity is Almost All You Need (for Prototypical-Part Models). WACV, 2026.