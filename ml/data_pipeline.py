"""Utilities for constructing datasets from pose estimation outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class PoseSample:
    video_id: str
    label: str
    joint_angles: dict[str, float]


def load_pose_samples(samples: Iterable[PoseSample]) -> pd.DataFrame:
    """Convert pose samples into a flat pandas DataFrame."""
    records = []
    for sample in samples:
        record = {"video_id": sample.video_id, "label": sample.label}
        record.update(sample.joint_angles)
        records.append(record)
    return pd.DataFrame.from_records(records)


def export_dataset(df: pd.DataFrame, destination: Path) -> None:
    """Persist the processed dataset to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)

def save_frame_angles(df: pd.DataFrame, shot_id: str, destination: Path) -> None:
    """
    Save pose angle data to CSV

    Document Schema:
    - shot_id
    - frame_number
    - elbow_angle
    - shoulder_angle
    - wrist_angle
    - etc.
    """

    df = df.copy()
    df.insert(0, "shot_id", shot_id)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)
    except Exception as e:
        print(f"Error saving frame angles to CSV: {e}")

def compute_joint_angles(pose_keypoints: list[float]) -> dict[str, float]:
    """Placeholder for vector math that converts raw keypoints into angles."""
    # TODO: implement vector math using NumPy for shoulder, elbow, hip, etc.
    return {"elbow_angle": 0.0, "knee_angle": 0.0}
