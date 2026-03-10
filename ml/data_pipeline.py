"""Utilities for constructing datasets from pose estimation outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

# I am making the assumption that we are using COCO17 keypoint format given what I see from pose_pipeline, so the mapping would look like:
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}
KEYPOINT_NAMES = {v: k for k, v in COCO_KEYPOINTS.items()}

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

# I change the parameter type for pose_keypoints to be a numpy array since keypoints should be in the format of (x, y) with the index mapped to the respective COCO joint.
def compute_joint_angles(pose_keypoints: np.ndarray, side:str = "right") -> dict[str, float]:
    """Placeholder for vector math that converts raw keypoints into angles."""
    # I am going to compute the following angles:
    #        - Shoulder
    #        - Knee 
    #        - elbow
    #        - wrist  <------ ! cannot be computed due to lack of hand keypoint tracking. 
    #        - hip

    if not isinstance(pose_keypoints, np.ndarray):
        raise TypeError(f"Expected pose_keypoints to be a numpy array, but got {type(pose_keypoints)}")
    
    if pose_keypoints.shape != (17, 2):
        raise ValueError(f"Expected shape (17, 2), got {pose_keypoints.shape}")  # expecting COCO17 keypoint format
    
    if side != "left" and side!= "right":
        raise ValueError("side paramter can only be 'left' or 'right'!")
    
    return {
        "elbow_angle": _compute_elbow_angle(pose_keypoints, side),
        "knee_angle": _compute_knee_angle(pose_keypoints, side),
        "shoulder_angle": _compute_shoulder_angle(pose_keypoints, side),
        "hip_angle": _compute_hip_angle(pose_keypoints, side)
    }

def _extract_point_2d(keypoints: np.ndarray, joint_name: str) -> np.ndarray:
    """
        responsible for extracting the x and y points from a specified joint from a keypoint.
    """
    if joint_name not in KEYPOINT_NAMES:
        return np.array([np.nana, np.nan], dtype=float)

    idx = KEYPOINT_NAMES[joint_name]
    try:
        point = keypoints[idx]
        x, y = float(point[0]), float(point[1])

        if x == 0.0 and y == 0.0:
            return np.array([np.nan, np.nan], dtype=float)
        
        return np.array([x, y], dtype=float)
    except (IndexError, ValueError, TypeError):
        return np.array([np.nan, np.nan], dtype=float)

def _angle_between(p1: np.ndarray, p2:np.ndarray, p3:np.ndarray) -> float:
    """
        Compute the angle between the two joints using vector math.
        The formula that I will leverage is the dot product peroperty:
        Assume we have vector A and B, 
            A dot B = |A| * |B| cos(theta), with theta being the angle between A and B.
        We know that a vector A and B can just be computed by taking the difference of point 1 and point 2, point 3 and point 2.

        Note: we must ensure that p2 is actually the middle joint, meaning that it's the joint of movement that we are most concerned of. 
            (e.g, for an arm, p2 must be elbow if we are concerned of the angle formed by the forearm and upperarm)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    n1 = np.linalg.norm(v1)   # norms are just he magnitude of the vectors
    n2 = np.linalg.norm(v2)

    if n1 == 0.0 or n2 == 0.0 or not np.isfinite(n1) or not np.isfinite(n2):
        return float("nan")

    theta = np.dot(v1, v2) / (n1 * n2)
    theta = np.clip(theta, -1.0, 1.0)

    theta_rad = np.arccos(theta)
    return float(np.degrees(theta_rad))

def _compute_elbow_angle(keypoints: np.ndarray, side:str) ->float:
    """Angle at elbow: shoulder -> elbow -> wrist"""
    if side != "right" and side != "left":
        return ValueError("side paramter must be either right or left!")

    shoulder = _extract_point_2d(keypoints, f"{side}_shoulder")
    elbow = _extract_point_2d(keypoints, f"{side}_elbow")
    wrist = _extract_point_2d(keypoints, f"{side}_wrist")
    return _angle_between(shoulder, elbow, wrist)

def _compute_knee_angle(keypoints: np.ndarray, side:str) -> float:
    """Knee angle: hip -> knee -> ankle"""
    if side != "right" and side != "left":
        raise ValueError("side paramter must be either right or left!")

    hip = _extract_point_2d(keypoints, f"{side}_hip")
    knee = _extract_point_2d(keypoints, f"{side}_knee")
    ankle = _extract_point_2d(keypoints, f"{side}_ankle")
    return _angle_between(hip, knee, ankle)


# !!! there are no keypoint tracking for hand, so wrist_angle cannot be properly computed.
# def _compute_wrist_angle(keypoints: np.ndarray, side:str) -> float:
#     """wrist angle: elbow -> wrist -> hand"""
#     if side != "right" and side != "left":
#         raise ValueError("side paramter must be either right or left!")

#     elbow = _extract_point_2d(keypoints, f"{side}_elbow")
#     wrist = _extract_point_2d(keypoints, f"{side}_wrist")
#     return _angle_between(hip, knee, ankle)

def _compute_shoulder_angle(keypoints: np.ndarray, side:str) -> float:
    """shoulder angle: hip -> shoulder -> elbow"""
    if side != "right" and side != "left":
        raise ValueError("side paramter must be either right or left!")

    hip = _extract_point_2d(keypoints, f"{side}_hip")
    shoulder = _extract_point_2d(keypoints, f"{side}_shoulder")
    elbow = _extract_point_2d(keypoints, f"{side}_elbow")
    return _angle_between(hip, shoulder, elbow)

def _compute_hip_angle(keypoints: np.ndarray, side:str) -> float:
    """hip angle: knee-> hip -> shoulder"""
    if side != "right" and side != "left":
        raise ValueError("side paramter must be either right or left!")
    knee = _extract_point_2d(keypoints, f"{side}_knee")
    hip = _extract_point_2d(keypoints, f"{side}_hip")
    shoulder = _extract_point_2d(keypoints, f"{side}_shoulder")
    return _angle_between(knee, hip, shoulder)

def link_features_labels():
    """Links Features and Labels together, handling any missing inputs for ML training"""
    
    # Rename files if csvs are in different location; additonally we can input at args if better
    df1 = pd.read_csv("'../data/raw/features.csv")
    df2 = pd.read_csv("../data/raw/labels.csv")
    
    # Ensure that both dataframes have the 'shot_id' column for merging
    if "shot_id" not in df1.columns or "shot_id" not in df2.columns:
        raise ValueError("Both CSV files must contain 'shot_id' column for merging.")
    
    # Only merge on the 'shot_id' column, using an inner join to keep only matching records
    # Any external records are not kept; if there are missing shot_id in either df, those records will be dropped
    df_merged = pd.merge(df1, df2, on="shot_id", how="inner")
    
    # Write the merged dataframe to a new CSV file; this will be the dataset used for ML training
    df_merged.to_csv("../data/processed/linked_features_labels.csv", index=False)
    
    return df_merged
    