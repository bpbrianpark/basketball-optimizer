"""Download an overlay image from the inference API, save it, and optionally display it."""
from __future__ import annotations
import argparse
import requests
import sys
from pathlib import Path
import numpy as np
import cv2

BASE_URL = "http://localhost:8000/api/inference"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch an overlay frame from the API, save to disk, and display it."
    )
    parser.add_argument("result_id", help="Result ID returned by /analyze")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--output", type=Path, default=None, help="Save path")
    parser.add_argument("--no-display", action="store_true", help="Skip OpenCV viewer")
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Inference API base URL (default: {BASE_URL})",
    )
    return parser.parse_args()


def fetch_overlay(base_url: str, result_id: str, frame: int) -> bytes:
    url = f"{base_url}/results/{result_id}/overlay"
    try:
        res = requests.get(url, params={"frame": frame}, timeout=30)
    except requests.ConnectionError:
        print("Error: Cannot connect. Is the server running?")
        sys.exit(1)
    except requests.Timeout:
        print("Error: Request timed out")
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"Error: Network request failed: {exc}")
        sys.exit(1)
    
    if res.status_code == 404:
        print(f"Error: Overlay not found (result_id={result_id}, frame={frame})")
        sys.exit(1)
    if res.status_code != 200:
        print(f"Error: API returned {res.status_code} - {res.text}")
        sys.exit(1)

    content_type = res.headers.get("content-type", "")
    if "image" not in content_type.lower():
        print(f"Error: Unexpected response content-type '{content_type}'")
        sys.exit(1)
    
    return res.content


def save_image(image_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    print(f"Saved to {path}")


def display_image(image_bytes: bytes) -> None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("Warning: Could not decode image.")
        return
    cv2.imshow("Overlay", img)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    image_bytes = fetch_overlay(args.base_url, args.result_id, args.frame)

    output = args.output or Path(f"overlay_{args.result_id}_frame{args.frame}.jpg")
    save_image(image_bytes, output)

    if not args.no_display:
        display_image(image_bytes)



if __name__ == "__main__":
    main()