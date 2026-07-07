# Experiment Protocol

## Project

Russian Vision-Language Model Adaptation using Open VK Datasets.

## Goal

Adapt a compact open Vision-Language Model to Russian-language visual
understanding tasks and evaluate whether fine-tuning on VK open datasets
improves both in-domain performance and cross-benchmark generalization.

## Base Model

Planned baseline candidate:

- Qwen/Qwen2.5-VL-3B-Instruct

The final base model choice may be revised only if technical experiments
show that the model cannot be used reliably in the available environment.

## Datasets

### GQA-ru

Source:

- deepvk/GQA-ru

Usage:

- `train_balanced_instructions`: source of training and validation questions
- `train_balanced_images`: corresponding training and validation images
- `testdev_balanced_instructions`: final in-domain evaluation questions
- `testdev_balanced_images`: corresponding final evaluation images

Questions are matched to images using:

`instruction.imageId == image.id`

### MMBench-ru

Source:

- deepvk/MMBench-ru

Usage:

- `dev`: cross-benchmark evaluation of multimodal generalization

MMBench-ru is not used for selecting checkpoints during the primary
GQA-ru fine-tuning experiment.

## GQA-ru Split Strategy

The original GQA-ru training set is divided into training and validation
subsets at the image level.

Reason:

Multiple questions may refer to the same image. A row-level random split
could place questions about the same image into both training and
validation sets, causing image leakage.

Split parameters:

- random seed: 42
- validation fraction: 0.05
- grouping key: `imageId`

Observed split:

- original training questions: 40,000
- generated training questions: 38,018
- generated validation questions: 1,982
- image overlap between generated train and validation: 0

The exact image IDs are stored in:

`data/splits/gqa_image_split.json`

## Final Evaluation Sets

### In-domain evaluation

GQA-ru testdev:

- 12,216 questions
- image overlap with GQA-ru train: 0

### Cross-benchmark evaluation

MMBench-ru dev:

- 3,910 examples
- invalid answer labels found during audit: 0

Available answer option counts:

- 4 options: 3,539 examples
- 3 options: 215 examples
- 2 options: 156 examples

Missing MMBench options represented as values such as `"nan"` must be
removed when constructing prompts.

## Experimental Stages

### E0: Base model

Evaluate the unmodified base VLM.

### E1: GQA-ru adaptation

Fine-tune the base model on the generated GQA-ru training split using
parameter-efficient fine-tuning.

Primary candidate:

- LoRA or QLoRA

Checkpoint selection uses only the generated GQA-ru validation split.

### E2: Mixed instruction tuning

Optional extended experiment using a mixture of:

- GQA-ru
- LLaVA-Instruct-ru

The purpose is to test whether broader Russian multimodal instruction
tuning improves cross-benchmark generalization.

## Evaluation Principles

1. No training on GQA-ru testdev.
2. No checkpoint selection using GQA-ru testdev.
3. No primary checkpoint selection using MMBench-ru dev.
4. All data splits must be deterministic and reproducible.
5. Base and fine-tuned models must use the same evaluation pipeline.
6. Prompt templates and decoding parameters must be recorded.
7. Evaluation results must include both aggregate and category-level metrics
   where supported by the dataset.

## Reproducibility

Current deterministic split seed:

`42`

Generated audit report:

`results/data_audit.json`

Generated GQA image-level split:

`data/splits/gqa_image_split.json`
