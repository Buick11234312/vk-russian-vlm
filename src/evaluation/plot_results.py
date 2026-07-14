"""Create result tables and figures for the VK Russian VLM project."""

from pathlib import Path
import csv
import matplotlib.pyplot as plt


OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = Path("results/summary_metrics.csv")

VALIDATION_RESULTS = [{"experiment": "Base", "train_examples": 0, "split": "validation", "accuracy": 0.4369}, {"experiment": "E2 LoRA", "train_examples": 2000, "split": "validation", "accuracy": 0.5721}, {"experiment": "E3 LoRA", "train_examples": 8000, "split": "validation", "accuracy": 0.5984}]
TESTDEV_RESULTS = [{"experiment": "Base", "train_examples": 0, "split": "testdev_2000", "accuracy": 0.4020}, {"experiment": "E1 LoRA", "train_examples": 512, "split": "testdev_2000", "accuracy": 0.4455}, {"experiment": "E3 LoRA", "train_examples": 8000, "split": "testdev_2000", "accuracy": 0.5265},]


def save_metrics_csv() -> None:
    rows = VALIDATION_RESULTS + TESTDEV_RESULTS
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "train_examples", "split", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def plot_line(results: list[dict], title: str, output_stem: str) -> None:
    x = [r["train_examples"] for r in results]
    y = [r["accuracy"] for r in results]
    labels = [r["experiment"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")

    for x_i, y_i, label in zip(x, y, labels):
        ax.annotate(f"{label}\n{y_i:.2%}", (x_i, y_i), textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Exact match accuracy")
    ax.set_ylim(0.35, 0.65)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / f"{output_stem}.png"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    save_metrics_csv()
    print(f"Saved {METRICS_CSV}")
    plot_line(VALIDATION_RESULTS, "GQA-ru held-out validation accuracy", "validation_accuracy_scaling")
    plot_line(TESTDEV_RESULTS, "GQA-ru testdev-2000 accuracy", "testdev_accuracy_scaling")


if __name__ == "__main__":
    main()
