from __future__ import annotations

import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
GQA_DATASET_ID = "deepvk/GQA-ru"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_gqa_example() -> tuple[object, dict]:
    """
    Load one real example from GQA-ru testdev and match
    the instruction to its image using imageId == id.
    """

    print("Loading one GQA-ru instruction...")

    instructions = load_dataset(
        GQA_DATASET_ID,
        "testdev_balanced_instructions",
        split="testdev",
        streaming=True,
    )

    instruction = next(iter(instructions))
    target_image_id = instruction["imageId"]

    print(f"Target image id: {target_image_id}")
    print(f"Question: {instruction['question']}")
    print(f"Gold answer: {instruction['answer']}")

    print("Searching for corresponding image...")

    images = load_dataset(
        GQA_DATASET_ID,
        "testdev_balanced_images",
        split="testdev",
        streaming=True,
    )

    for image_row in images:
        if image_row["id"] == target_image_id:
            return image_row["image"], instruction

    raise RuntimeError(
        f"Image {target_image_id!r} was not found."
    )


def load_model(
    device: torch.device,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    print()
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    model_dtype = (
        torch.float16
        if device.type == "mps"
        else torch.float32
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    )

    model.to(device)
    model.eval()

    return model, processor


def generate_answer(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image: object,
    question: str,
    device: torch.device,
) -> str:
    prompt = (
        f"{question}\n"
        "Ответь одним словом."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
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
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }

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

    answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return answer.strip()


def main() -> int:
    device = get_device()

    if device.type != "mps":
        print(
            "WARNING: MPS is unavailable. "
            "Inference will run on CPU.",
            file=sys.stderr,
        )

    image, instruction = load_gqa_example()

    print(
        f"Image size: {getattr(image, 'size', 'unknown')}"
    )

    model, processor = load_model(device)

    print()
    print("Running inference...")

    prediction = generate_answer(
        model=model,
        processor=processor,
        image=image,
        question=instruction["question"],
        device=device,
    )

    print()
    print("=" * 72)
    print("SMOKE TEST RESULT")
    print("=" * 72)
    print(f"Question:    {instruction['question']}")
    print(f"Gold answer: {instruction['answer']}")
    print(f"Prediction:  {prediction}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
