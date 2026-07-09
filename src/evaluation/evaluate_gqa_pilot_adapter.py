from __future__ import annotations

import json
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_ID = "deepvk/GQA-ru"
ADAPTER_DIR = Path("outputs/pilot_lora")

SAMPLE_PATH = Path("data/splits/gqa_baseline_sample.json")
BASELINE_SUMMARY_PATH = Path("results/baseline/gqa_summary.json")
OUTPUT_PATH = Path("results/pilot/gqa_predictions.jsonl")
SUMMARY_PATH = Path("results/pilot/gqa_summary.json")


def normalize_exact_match(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[«»„“”–—…]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_group_accuracy(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in records:
        key = str(record[field])
        grouped[key]["total"] += 1
        grouped[key]["correct"] += int(record["correct"])

    result = {}

    for key, values in sorted(grouped.items()):
        total = values["total"]
        correct = values["correct"]
        result[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }

    return result


def load_fixed_instructions() -> list[dict[str, Any]]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample file not found: {SAMPLE_PATH}"
        )

    sample_data = json.loads(
        SAMPLE_PATH.read_text(encoding="utf-8")
    )
    question_ids = sample_data["question_ids"]

    print(
        f"Loading fixed sample of {len(question_ids)} question IDs..."
    )

    dataset = load_dataset(
        DATASET_ID,
        "testdev_balanced_instructions",
        split="testdev",
    )

    row_by_id = {row["id"]: dict(row) for row in dataset}

    missing_ids = [
        question_id
        for question_id in question_ids
        if question_id not in row_by_id
    ]

    if missing_ids:
        raise RuntimeError(
            f"Missing question IDs: {missing_ids[:10]}"
        )

    return [row_by_id[qid] for qid in question_ids]


def load_images_for_instructions(
    instructions: list[dict[str, Any]],
) -> dict[str, Any]:
    target_image_ids = {
        row["imageId"] for row in instructions
    }

    print(f"Need {len(target_image_ids)} unique images.")

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
            f"Missing image IDs: {sorted(missing_ids)[:10]}"
        )

    return image_by_id


def load_model_and_processor() -> tuple[PeftModel, AutoProcessor]:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}"
        )

    print()
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    print(f"Loading base model: {MODEL_ID}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Loading adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.to("cuda")
    model.eval()

    return model, processor


def generate_answer(
    model: PeftModel,
    processor: AutoProcessor,
    image: Any,
    question: str,
) -> str:
    prompt = f"{question}\nОтветь одним словом."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
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
        key: value.to("cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, prompt_length:]

    prediction = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return prediction.strip()


def main() -> None:
    print("=" * 80)
    print("GQA PILOT ADAPTER EVALUATION")
    print("=" * 80)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    instructions = load_fixed_instructions()
    image_by_id = load_images_for_instructions(instructions)
    model, processor = load_model_and_processor()

    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(instructions, start=1):
            example_started_at = time.perf_counter()

            prediction = generate_answer(
                model=model,
                processor=processor,
                image=image_by_id[row["imageId"]],
                question=row["question"],
            )

            gold_normalized = normalize_exact_match(row["answer"])
            prediction_normalized = normalize_exact_match(prediction)
            is_correct = gold_normalized == prediction_normalized

            latency = time.perf_counter() - example_started_at

            record = {
                "id": row["id"],
                "image_id": row["imageId"],
                "question": row["question"],
                "gold": row["answer"],
                "prediction": prediction,
                "gold_normalized": gold_normalized,
                "prediction_normalized": prediction_normalized,
                "correct": is_correct,
                "detailed_type": row["types"]["detailed"],
                "semantic_type": row["types"]["semantic"],
                "structural_type": row["types"]["structural"],
                "latency_seconds": latency,
                "model_id": MODEL_ID,
                "adapter_dir": str(ADAPTER_DIR),
            }

            records.append(record)
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            correct_so_far = sum(
                int(item["correct"]) for item in records
            )
            accuracy_so_far = correct_so_far / len(records)

            print(
                f"[{index:>3}/{len(instructions)}] "
                f"acc={accuracy_so_far:.3f} "
                f"time={latency:.2f}s "
                f"gold={row['answer']!r} "
                f"pred={prediction!r}"
            )

    total_elapsed = time.perf_counter() - started_at
    total = len(records)
    correct = sum(int(record["correct"]) for record in records)
    accuracy = correct / total if total else 0.0
    mean_latency = total_elapsed / total if total else 0.0

    baseline_accuracy = None

    if BASELINE_SUMMARY_PATH.exists():
        baseline_summary = json.loads(
            BASELINE_SUMMARY_PATH.read_text(encoding="utf-8")
        )
        baseline_accuracy = float(
            baseline_summary["accuracy"]
        )

    accuracy_delta = (
        accuracy - baseline_accuracy
        if baseline_accuracy is not None
        else None
    )

    summary = {
        "model_id": MODEL_ID,
        "adapter_dir": str(ADAPTER_DIR),
        "sample_path": str(SAMPLE_PATH),
        "examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_delta": accuracy_delta,
        "total_time_seconds": total_elapsed,
        "mean_latency_seconds": mean_latency,
        "by_detailed_type": compute_group_accuracy(
            records,
            "detailed_type",
        ),
        "by_semantic_type": compute_group_accuracy(
            records,
            "semantic_type",
        ),
        "by_structural_type": compute_group_accuracy(
            records,
            "structural_type",
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("GQA PILOT ADAPTER EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Examples:          {total}")
    print(f"Correct:           {correct}")
    print(f"Accuracy:          {accuracy:.4f}")

    if baseline_accuracy is not None:
        print(
            f"Baseline accuracy: "
            f"{baseline_accuracy:.4f}"
        )
        print(
            f"Accuracy delta:    "
            f"{accuracy_delta:+.4f}"
        )

    print(f"Total time:        {total_elapsed:.2f}s")
    print(f"Mean latency:      {mean_latency:.2f}s/example")
    print(f"Predictions:       {OUTPUT_PATH}")
    print(f"Summary:           {SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
