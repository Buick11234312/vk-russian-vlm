from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = "outputs/smoke_lora"
DATASET_ID = "deepvk/GQA-ru"


def load_one_example() -> tuple[Any, dict]:
    instructions = load_dataset(
        DATASET_ID,
        "train_balanced_instructions",
        split="train",
        streaming=True,
    )
    instruction = next(iter(instructions))
    target_image_id = instruction["imageId"]

    images = load_dataset(
        DATASET_ID,
        "train_balanced_images",
        split="train",
        streaming=True,
    )

    for row in images:
        if row["id"] == target_image_id:
            return row["image"], instruction

    raise RuntimeError(
        f"Image {target_image_id!r} was not found."
    )


def main() -> None:
    print("=" * 80)
    print("LORA ADAPTER RELOAD TEST")
    print("=" * 80)

    image, instruction = load_one_example()

    print(f"Question: {instruction['question']}")
    print(f"Gold:     {instruction['answer']}")

    processor = AutoProcessor.from_pretrained(
        ADAPTER_DIR,
    )

    print()
    print("Loading base model...")

    base_model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.to("cuda")
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{instruction['question']}\n"
                        "Ответь одним словом."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to("cuda")
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }

    print("Running inference...")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]

    generated_only = generated_ids[
        :,
        prompt_length:,
    ]

    prediction = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print()
    print("=" * 80)
    print("RELOAD TEST PASSED")
    print("=" * 80)
    print(f"Question:   {instruction['question']}")
    print(f"Gold:       {instruction['answer']}")
    print(f"Prediction: {prediction}")
    print("=" * 80)


if __name__ == "__main__":
    main()
