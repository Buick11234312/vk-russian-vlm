from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "deepvk/GQA-ru"

SPLIT_PATH = Path("data/splits/gqa_image_split.json")
SAMPLE_PATH = Path("data/splits/gqa_train_8000_sample.json")
OUTPUT_DIR = Path("outputs/train_8000_completion_only")
METRICS_PATH = OUTPUT_DIR / "train_metrics.json"

SEED = 42
NUM_TRAIN_EXAMPLES = 8000

PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
PER_DEVICE_EVAL_BATCH_SIZE = 1

NUM_EPOCHS = 1.0
LEARNING_RATE = 1e-4


def sample_train_rows(
    rows: list[dict[str, Any]],
    allowed_image_ids: set[str],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["imageId"] in allowed_image_ids
    ]

    if len(eligible) < n:
        raise RuntimeError(
            f"Need {n} train rows, found only {len(eligible)}."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(eligible)), k=n)

    return [
        eligible[index]
        for index in indices
    ]


def load_selected_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    split_data = json.loads(
        SPLIT_PATH.read_text(encoding="utf-8")
    )

    train_image_ids = set(
        split_data["train_image_ids"]
    )
    validation_image_ids = set(
        split_data["validation_image_ids"]
    )

    split_overlap = (
        train_image_ids
        & validation_image_ids
    )

    if split_overlap:
        raise RuntimeError(
            f"Image leakage in split file: "
            f"{len(split_overlap)} overlapping image IDs."
        )

    print("Loading GQA-ru train instructions...")

    instructions = load_dataset(
        DATASET_ID,
        "train_balanced_instructions",
        split="train",
    )

    rows = [
        dict(row)
        for row in instructions
    ]

    train_rows = sample_train_rows(
        rows=rows,
        allowed_image_ids=train_image_ids,
        n=NUM_TRAIN_EXAMPLES,
        seed=SEED,
    )

    # Use the full held-out validation partition:
    # every question whose image belongs to validation_image_ids.
    validation_rows = [
        row
        for row in rows
        if row["imageId"] in validation_image_ids
    ]

    train_selected_images = {
        row["imageId"]
        for row in train_rows
    }

    validation_selected_images = {
        row["imageId"]
        for row in validation_rows
    }

    selected_overlap = (
        train_selected_images
        & validation_selected_images
    )

    if selected_overlap:
        raise RuntimeError(
            "Train/validation image leakage detected."
        )

    if len(validation_rows) != 1982:
        raise RuntimeError(
            f"Expected 1982 validation rows, "
            f"found {len(validation_rows)}."
        )

    SAMPLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_PATH.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "seed": SEED,
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "train_question_ids": [
                    row["id"]
                    for row in train_rows
                ],
                "validation_question_ids": [
                    row["id"]
                    for row in validation_rows
                ],
                "train_unique_images": len(
                    train_selected_images
                ),
                "validation_unique_images": len(
                    validation_selected_images
                ),
                "image_overlap": len(
                    selected_overlap
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return train_rows, validation_rows


def load_images(
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_image_ids = {
        row["imageId"]
        for row in selected_rows
    }

    print(
        f"Need {len(target_image_ids)} unique images."
    )

    images_stream = load_dataset(
        DATASET_ID,
        "train_balanced_images",
        split="train",
        streaming=True,
    )

    image_by_id: dict[str, Any] = {}

    for row in images_stream:
        image_id = row["id"]

        if image_id in target_image_ids:
            image_by_id[image_id] = row["image"]

            found = len(image_by_id)
            total = len(target_image_ids)

            if (
                found == 1
                or found % 100 == 0
                or found == total
            ):
                print(
                    f"\rFound images: {found}/{total}",
                    end="",
                    flush=True,
                )

            if found == total:
                break

    print()

    missing = (
        target_image_ids
        - set(image_by_id)
    )

    if missing:
        raise RuntimeError(
            f"Missing image IDs: "
            f"{sorted(missing)[:20]}"
        )

    return image_by_id


def to_vlm_dataset(
    rows: list[dict[str, Any]],
    image_by_id: dict[str, Any],
) -> Dataset:
    examples = []

    for row in rows:
        examples.append(
            {
                "image": image_by_id[
                    row["imageId"]
                ],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"{row['question']}\n"
                                    "Ответь одним словом."
                                ),
                            },
                        ],
                    }
                ],
                "completion": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": row["answer"],
                            }
                        ],
                    }
                ],
            }
        )

    return Dataset.from_list(examples)


def main() -> None:
    print("=" * 80)
    print("GQA E3 TRAINING - 8000 EXAMPLES - COMPLETION ONLY")
    print("=" * 80)

    train_rows, validation_rows = load_selected_rows()

    image_by_id = load_images(
        train_rows + validation_rows
    )

    train_dataset = to_vlm_dataset(
        train_rows,
        image_by_id,
    )

    validation_dataset = to_vlm_dataset(
        validation_rows,
        image_by_id,
    )

    effective_batch_size = (
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    expected_optimizer_steps = math.ceil(
        len(train_dataset)
        / effective_batch_size
    )

    warmup_steps = max(
        1,
        round(
            expected_optimizer_steps * 0.05
        ),
    )

    print()
    print(
        f"Train examples:             "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation examples:        "
        f"{len(validation_dataset)}"
    )
    print(
        f"Train batch size/device:    "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE}"
    )
    print(
        f"Gradient accumulation:      "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )
    print(
        f"Effective train batch size: "
        f"{effective_batch_size}"
    )
    print(
        f"Expected optimizer steps:   "
        f"{expected_optimizer_steps}"
    )
    print(
        f"Warmup steps:               "
        f"{warmup_steps}"
    )
    print(
        f"Sample IDs saved to:        "
        f"{SAMPLE_PATH}"
    )

    print()
    print(f"Loading processor: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    processor.tokenizer.padding_side = "right"

    print()
    print(f"Loading model: {MODEL_ID}")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj)$"
        ),
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        max_length=None,
        completion_only_loss=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eos_token="<|im_end|>",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    print()
    print("=" * 80)
    print("TRAINABLE PARAMETERS")
    print("=" * 80)

    trainer.model.print_trainable_parameters()

    trainable_names = [
        name
        for name, parameter
        in trainer.model.named_parameters()
        if parameter.requires_grad
    ]

    visual_trainable = [
        name
        for name in trainable_names
        if "visual" in name
    ]

    if visual_trainable:
        raise RuntimeError(
            "Vision parameters unexpectedly trainable:\n"
            + "\n".join(
                visual_trainable[:20]
            )
        )

    print(
        "Vision trainable parameters: 0"
    )

    print()
    print("=" * 80)
    print("PRE-TRAIN VALIDATION LOSS")
    print("=" * 80)

    pre_eval = trainer.evaluate()
    pre_eval_loss = float(
        pre_eval["eval_loss"]
    )

    print(
        f"Pre-train eval loss: "
        f"{pre_eval_loss:.6f}"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print("=" * 80)
    print("STARTING E3 TRAINING")
    print("=" * 80)

    started_at = time.perf_counter()

    train_result = trainer.train()

    train_seconds = (
        time.perf_counter()
        - started_at
    )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print()
    print("=" * 80)
    print("POST-TRAIN VALIDATION LOSS")
    print("=" * 80)

    post_eval = trainer.evaluate()

    post_eval_loss = float(
        post_eval["eval_loss"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    processor.save_pretrained(
        str(OUTPUT_DIR)
    )

    metrics = {
        "experiment": "E3",
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "train_examples": len(
            train_dataset
        ),
        "validation_examples": len(
            validation_dataset
        ),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": (
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "effective_train_batch_size": (
            effective_batch_size
        ),
        "expected_optimizer_steps": (
            expected_optimizer_steps
        ),
        "global_step": int(
            train_result.global_step
        ),
        "learning_rate": LEARNING_RATE,
        "warmup_steps": warmup_steps,
        "train_loss": float(
            train_result.training_loss
        ),
        "pre_train_eval_loss": (
            pre_eval_loss
        ),
        "post_train_eval_loss": (
            post_eval_loss
        ),
        "eval_loss_change": (
            post_eval_loss
            - pre_eval_loss
        ),
        "train_runtime_seconds": (
            train_seconds
        ),
        "peak_cuda_memory_gb": (
            peak_memory_gb
        ),
        "trainable_parameters": (
            3_686_400
        ),
        "vision_trainable_parameters": 0,
        "loss_scope": "completion_only",
        "sample_path": str(
            SAMPLE_PATH
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("E3 TRAINING COMPLETE")
    print("=" * 80)
    print(
        f"Global step:          "
        f"{train_result.global_step}"
    )
    print(
        f"Train loss:           "
        f"{train_result.training_loss:.6f}"
    )
    print(
        f"Pre-train eval loss:  "
        f"{pre_eval_loss:.6f}"
    )
    print(
        f"Post-train eval loss: "
        f"{post_eval_loss:.6f}"
    )
    print(
        f"Eval loss change:     "
        f"{post_eval_loss - pre_eval_loss:+.6f}"
    )
    print(
        f"Train runtime:        "
        f"{train_seconds:.2f}s"
    )
    print(
        f"Peak CUDA memory:     "
        f"{peak_memory_gb:.2f} GB"
    )
    print(
        f"Adapter saved to:     "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Metrics saved to:     "
        f"{METRICS_PATH}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
