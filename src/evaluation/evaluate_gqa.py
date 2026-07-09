from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "deepvk/GQA-ru"

DEFAULT_OUTPUT_PATH = Path(
    "results/baseline/gqa_predictions.jsonl"
)

DEFAULT_SUMMARY_PATH = Path(
    "results/baseline/gqa_summary.json"
)

DEFAULT_SAMPLE_PATH = Path(
    "data/splits/gqa_baseline_sample.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )

    parser.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
    )

    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_exact_match(text: str) -> str:
    """
    Exact-match normalization aligned with the GQA-ru
    benchmark configuration:

    - ignore case
    - ignore punctuation
    - normalize whitespace
    """

    text = text.lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    # Extra punctuation commonly present in Russian text.
    text = re.sub(
        r"[«»„“”–—…]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def load_random_instructions(
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    print("Loading GQA-ru testdev instructions...")

    dataset = load_dataset(
        DATASET_ID,
        "testdev_balanced_instructions",
        split="testdev",
    )

    if limit > len(dataset):
        raise ValueError(
            f"--limit={limit} exceeds dataset size "
            f"{len(dataset)}."
        )

    rng = random.Random(seed)

    indices = sorted(
        rng.sample(
            range(len(dataset)),
            k=limit,
        )
    )

    return [
        dataset[index]
        for index in indices
    ]


def load_images_for_instructions(
    instructions: list[dict[str, Any]],
) -> dict[str, Any]:
    target_image_ids = {
        row["imageId"]
        for row in instructions
    }

    print(
        f"Need {len(target_image_ids)} unique images."
    )

    images_stream = load_dataset(
        DATASET_ID,
        "testdev_balanced_images",
        split="testdev",
        streaming=True,
    )

    image_by_id: dict[str, Any] = {}

    for row in images_stream:
        image_id = row["id"]

        if image_id in target_image_ids:
            image_by_id[image_id] = row["image"]

            print(
                f"\rFound images: "
                f"{len(image_by_id)}/"
                f"{len(target_image_ids)}",
                end="",
                flush=True,
            )

            if len(image_by_id) == len(target_image_ids):
                break

    print()

    missing_ids = (
        target_image_ids
        - set(image_by_id)
    )

    if missing_ids:
        raise RuntimeError(
            "Missing image IDs: "
            f"{sorted(missing_ids)[:10]}"
        )

    return image_by_id


def load_model(
    device: torch.device,
) -> tuple[
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
]:
    print()
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    dtype = (
        torch.float16
        if device.type == "mps"
        else torch.float32
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            dtype=dtype,
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
    image: Any,
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
            num_beams=1,
        )

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    generated_only = generated_ids[
        :,
        prompt_length:,
    ]

    prediction = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return prediction.strip()


def compute_group_accuracy(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    for record in records:
        key = str(record[field])

        grouped[key]["total"] += 1
        grouped[key]["correct"] += int(
            record["correct"]
        )

    result = {}

    for key, values in sorted(
        grouped.items()
    ):
        total = values["total"]
        correct = values["correct"]

        result[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }

    return result


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError(
            "--limit must be positive."
        )

    for path in (
        args.output,
        args.summary,
        args.sample_output,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    device = get_device()

    instructions = load_random_instructions(
        limit=args.limit,
        seed=args.seed,
    )

    args.sample_output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "limit": args.limit,
                "question_ids": [
                    row["id"]
                    for row in instructions
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    image_by_id = load_images_for_instructions(
        instructions
    )

    model, processor = load_model(
        device=device,
    )

    records: list[dict[str, Any]] = []

    started_at = time.perf_counter()

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for index, row in enumerate(
            instructions,
            start=1,
        ):
            example_started_at = (
                time.perf_counter()
            )

            prediction = generate_answer(
                model=model,
                processor=processor,
                image=image_by_id[row["imageId"]],
                question=row["question"],
                device=device,
            )

            gold_normalized = (
                normalize_exact_match(
                    row["answer"]
                )
            )

            prediction_normalized = (
                normalize_exact_match(
                    prediction
                )
            )

            is_correct = (
                gold_normalized
                == prediction_normalized
            )

            latency = (
                time.perf_counter()
                - example_started_at
            )

            record = {
                "id": row["id"],
                "image_id": row["imageId"],
                "question": row["question"],
                "gold": row["answer"],
                "prediction": prediction,
                "gold_normalized": (
                    gold_normalized
                ),
                "prediction_normalized": (
                    prediction_normalized
                ),
                "correct": is_correct,
                "detailed_type": (
                    row["types"]["detailed"]
                ),
                "semantic_type": (
                    row["types"]["semantic"]
                ),
                "structural_type": (
                    row["types"]["structural"]
                ),
                "latency_seconds": latency,
                "model_id": MODEL_ID,
            }

            records.append(record)

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            correct_so_far = sum(
                int(item["correct"])
                for item in records
            )

            accuracy_so_far = (
                correct_so_far
                / len(records)
            )

            print(
                f"[{index:>3}/{len(instructions)}] "
                f"acc={accuracy_so_far:.3f} "
                f"time={latency:.2f}s "
                f"gold={row['answer']!r} "
                f"pred={prediction!r}"
            )

    total_elapsed = (
        time.perf_counter()
        - started_at
    )

    total = len(records)

    correct = sum(
        int(record["correct"])
        for record in records
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    mean_latency = (
        total_elapsed / total
        if total
        else 0.0
    )

    estimated_full_seconds = (
        mean_latency * 12216
    )

    summary = {
        "model_id": MODEL_ID,
        "device": str(device),
        "seed": args.seed,
        "examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_time_seconds": total_elapsed,
        "mean_latency_seconds": mean_latency,
        "estimated_full_eval_hours": (
            estimated_full_seconds / 3600
        ),
        "by_detailed_type": (
            compute_group_accuracy(
                records,
                "detailed_type",
            )
        ),
        "by_semantic_type": (
            compute_group_accuracy(
                records,
                "semantic_type",
            )
        ),
        "by_structural_type": (
            compute_group_accuracy(
                records,
                "structural_type",
            )
        ),
    }

    args.summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("GQA BASELINE EVALUATION")
    print("=" * 80)
    print(f"Model:          {MODEL_ID}")
    print(f"Device:         {device}")
    print(f"Seed:           {args.seed}")
    print(f"Examples:       {total}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Total time:     {total_elapsed:.2f}s")
    print(f"Mean latency:   {mean_latency:.2f}s/example")
    print(
        "Estimated full: "
        f"{estimated_full_seconds / 3600:.2f}h"
    )
    print(f"Predictions:    {args.output}")
    print(f"Summary:        {args.summary}")
    print(f"Sample IDs:     {args.sample_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
