from __future__ import annotations

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

NUM_EXAMPLES = 16
OUTPUT_DIR = "outputs/smoke_lora"


def load_smoke_dataset() -> Dataset:
    print(f"Loading {NUM_EXAMPLES} GQA-ru instructions...")

    instructions_stream = load_dataset(
        DATASET_ID,
        "train_balanced_instructions",
        split="train",
        streaming=True,
    )

    instructions = []

    for row in instructions_stream:
        instructions.append(row)

        if len(instructions) >= NUM_EXAMPLES:
            break

    target_image_ids = {
        row["imageId"]
        for row in instructions
    }

    print(f"Need {len(target_image_ids)} unique images.")

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

    missing_ids = target_image_ids - set(image_by_id)

    if missing_ids:
        raise RuntimeError(
            f"Missing images: {sorted(missing_ids)}"
        )

    examples = []

    for row in instructions:
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

    dataset = Dataset.from_list(examples)

    print()
    print("Dataset features:")
    print(dataset.features)

    print()
    print("First question:")
    print(instructions[0]["question"])

    print("First answer:")
    print(instructions[0]["answer"])

    return dataset


def main() -> None:
    print("=" * 80)
    print("LORA TRAINING SMOKE TEST")
    print("=" * 80)

    dataset = load_smoke_dataset()

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
        output_dir=OUTPUT_DIR,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        max_length=None,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        eos_token="<|im_end|>",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
        for name, parameter in trainer.model.named_parameters()
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

    print()
    print("Vision trainable parameters: 0")

    print()
    print("=" * 80)
    print("STARTING 2 TRAINING STEPS")
    print("=" * 80)

    result = trainer.train()

    print()
    print("=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print(f"Global step: {result.global_step}")
    print(f"Training loss: {result.training_loss}")

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print()
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
