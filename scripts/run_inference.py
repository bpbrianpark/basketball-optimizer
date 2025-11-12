"""CLI entry point for running pose analysis on a local video file."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.services.pose_estimator import PoseEstimatorService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a basketball shooting video")
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "--model", type=Path, default=None, help="Optional path to a custom YOLOv8 pose model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PoseEstimatorService(model_path=str(args.model) if args.model else None)

    result = service.analyze_video(args.video)

    print("Score:", result.score)
    print("Strengths:")
    for item in result.strengths:
        print(" -", item)
    print("Weaknesses:")
    for item in result.weaknesses:
        print(" -", item)
    print("Metadata:")
    for key, value in result.metadata.items():
        print(f" - {key}: {value}")


if __name__ == "__main__":
    main()
