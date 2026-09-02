from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meteo_ml.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG-style assistant for project scenario products.")
    parser.add_argument(
        "question",
        nargs="?",
        default="How should users interpret heavy-rain scenario indicators?",
        help="Question about the rainfall or climate-scenario prototype.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(answer_question(args.question, root))


if __name__ == "__main__":
    main()
