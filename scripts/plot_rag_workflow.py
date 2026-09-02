from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meteo-ml-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, width, height, title, body, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=color,
        edgecolor="#1f2937",
        linewidth=1.2,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=11, weight="bold", color="#111827")
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", fontsize=9, color="#374151")


def add_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#374151"))


def main() -> None:
    output = Path("figures/rag_scenario_assistant_workflow.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    add_box(ax, (0.5, 3.6), 2.2, 1.15, "User question", "Example:\nWhat changes in\nheavy-rain risk?", "#fef3c7")
    add_box(ax, (3.35, 3.6), 2.2, 1.15, "Project corpus", "README, docs,\nscenario summary", "#dbeafe")
    add_box(ax, (6.2, 3.6), 2.2, 1.15, "Retriever", "TF-IDF ranks\nrelevant chunks", "#dcfce7")
    add_box(ax, (9.05, 3.6), 2.45, 1.15, "Cited answer", "Evidence snippets\nwith source files", "#fce7f3")

    add_box(ax, (3.1, 1.15), 2.6, 1.05, "Scenario products", "CSV + Markdown\nextreme indicators", "#e0f2fe")
    add_box(ax, (6.4, 1.15), 2.6, 1.05, "Responsible GenAI layer", "LLM can summarize\nretrieved evidence", "#ede9fe")

    add_arrow(ax, (2.7, 4.18), (3.35, 4.18))
    add_arrow(ax, (5.55, 4.18), (6.2, 4.18))
    add_arrow(ax, (8.4, 4.18), (9.05, 4.18))
    add_arrow(ax, (4.4, 2.2), (4.4, 3.6))
    add_arrow(ax, (7.7, 2.2), (7.55, 3.6))
    add_arrow(ax, (9.0, 1.7), (9.95, 3.6))

    ax.text(6, 5.45, "GenAI/RAG Extension for Climate Scenario Product Support", ha="center", fontsize=16, weight="bold")
    ax.text(
        6,
        0.35,
        "Local demo: retrieval from project files. Production extension: embeddings, LLM generation, user feedback, evaluation, and access-controlled knowledge base.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )

    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
