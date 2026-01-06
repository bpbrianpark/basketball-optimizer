import argparse
import os
import sys
import cv2
import pandas as pd

from models.pose_pipeline import PosePipeline
from scripts.score_data import score_shot

def save_first_frame(video_path, output_name="first_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_name, frame)
        print(f"Successfully saved overlay frame to: {output_name}")
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Pose Analysis and Scoring CLI")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: File '{args.video_path}' not found.")
        sys.exit(1)

    pipeline = PosePipeline()
    results_df = pipeline.process_video(args.video_path)

    score, feedback = score_shot(results_df)
    print(f"\n--- Analysis Results ---")
    print(f"Final Score: {score:.2f}")
    print(f"Feedback: {feedback}\n")

    save_first_frame(args.video_path)

if __name__ == "__main__":
    main()