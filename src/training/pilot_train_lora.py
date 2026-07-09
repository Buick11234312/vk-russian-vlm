from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "deepvk/GQA-ru"

SPLIT_PATH = Path("data/splits/gqa_image_split.json")
SAMPLE_PATH = Path("data/splits/gqa_pilot_sample.json")
OUTPUT_DIR = Path("outputs/pilot_lora")
METRICS_PATH = OUTPUT_DIR / "pilot_metrics.json"

SEED = 42
NUM_TRAIN_EXAMPLES = 512
NUM_VALIDATION_EXAMPLES = 128


def sample_rows(
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
            f"Need {n} eligible rows, found only {len(eligible)}."
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

    overlap = train_image_ids & validation_image_ids
    if overlap:
        raise RuntimeError(
            f"Image leakage in split file: {len(overlap)} overlaps."
        )

    print("Loading GQA-ru train instructions...")

    instructions = load_dataset(
        DATASET_ID,
        "train_balanced_instructions",
        split="train",
    )

    rows = [dict(row) for row in instructions]

    train_rows = sample_rows(
        rows=rows,
        allowed_image_ids=train_image_ids,
        n=NUM_TRAIN_EXAMPLES,
        seed=SEED,
    )

    validation_rows = sample_rows(
        rows=rows,
        allowed_image_ids=validation_image_ids,
        n=NUM_VALIDATION_EXAMPLES,
        seed=SEED + 1,
    )

    train_selected_images = {
        row["imageId"] for row in train_rows
    }
    validation_selected_images = {
        row["imageId"] for row in validation_rows
    }

    selected_overlap = (
        train_selected_images
        & validation_selected_images
    )

    if selected_overlap:
        raise RuntimeError(
            "Pilot train/validation image leakage detected."
        )

    SAMPLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "train_examples": NUM_TRAIN_EXAMPLES,
                "validation_examples": NUM_VALIDATION_EXAMPLES,
                "train_question_ids": [
                    row["id"] for row in train_rows
                ],
                "validation_question_ids": [
                    row["id"] for row in validation_rows
                ],
                "train_unique_images": len(train_selected_images),
                "validation_unique_images": len(
                    validation_selected_images
                ),
                "image_overlap": len(selected_overlap),
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

            print(
                f"\rFound images: "
                f"{len(image_by_id)}/{len(target_image_ids)}",
                end="",
                flush=True,
            )

            if len(image_by_id) == len(target_image_ids):
                break

    print()

    missing = target_image_ids - set(image_by_id)
    if missing:
        raise RuntimeError(
            f"Missing image IDs: {sorted(missing)[:20]}"
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
                "image": image_by_id[row["imageId"]],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": (
                                    f"{row['question']}\n"
                                    "Ответь одним словом."
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": row["answer"],
                            }
                        ],
                    },
                ],
            }
        )

    return Dataset.from_list(examples)


def main() -> None:
    print("=" * 80)
    print("GQA PILOT LORA TRAINING")
    print("=" * 80)

    train_rows, validation_rows = load_selected_rows()

    all_selected_rows = train_rows + validation_rows
    image_by_id = load_images(all_selected_rows)

    train_dataset = to_vlm_dataset(
        train_rows,
        image_by_id,
    )

    validation_dataset = to_vlm_dataset(
        validation_rows,
        image_by_id,
    )

    print()
    print(f"Train examples:      {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Sample IDs saved to: {SAMPLE_PATH}")

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
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        max_length=None,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=8,
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
            + "\n".join(visual_trainable[:20])
        )

    print("Vision trainable parameters: 0")

    print()
    print("=" * 80)
    print("PRE-TRAIN VALIDATION")
    print("=" * 80)

    pre_eval = trainer.evaluate()
    pre_eval_loss = float(pre_eval["eval_loss"])

    print(f"Pre-train eval loss: {pre_eval_loss:.6f}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print("=" * 80)
    print("STARTING PILOT TRAINING")
    print("=" * 80)

    started_at = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started_at

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print()
    print("=" * 80)
    print("POST-TRAIN VALIDATION")
    print("=" * 80)

    post_eval = trainer.evaluate()
    post_eval_loss = float(post_eval["eval_loss"])

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    metrics = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "global_step": int(train_result.global_step),
        "train_loss": float(train_result.training_loss),
        "pre_train_eval_loss": pre_eval_loss,
        "post_train_eval_loss": post_eval_loss,
        "eval_loss_change": (
            post_eval_loss - pre_eval_loss
        ),
        "train_runtime_seconds": train_seconds,
        "peak_cuda_memory_gb": peak_memory_gb,
        "trainable_parameters": 3_686_400,
        "vision_trainable_parameters": 0,
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
    print("PILOT TRAINING COMPLETE")
    print("=" * 80)
    print(f"Global step:          {train_result.global_step}")
    print(f"Train loss:           {train_result.training_loss:.6f}")
    print(f"Pre-train eval loss:  {pre_eval_loss:.6f}")
    print(f"Post-train eval loss: {post_eval_loss:.6f}")
    print(
        f"Eval loss change:     "
        f"{post_eval_loss - pre_eval_loss:+.6f}"
    )
    print(f"Train runtime:        {train_seconds:.2f}s")
    print(f"Peak CUDA memory:     {peak_memory_gb:.2f} GB")
    print(f"Adapter saved to:     {OUTPUT_DIR}")
    print(f"Metrics saved to:     {METRICS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
