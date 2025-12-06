import json
from pathlib import Path
from typing import Dict, Any

class InferenceService:
    def __init__(self, data_dir: str = "data/inference"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.videos_path = self.data_dir / "videos.json"
        self.results_path = self.data_dir / "results.json"

        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)

        self.video_store: Dict[str, str] = self._load_store(self.videos_path)
        self.results_store: Dict[str, Any] = self._load_store(self.results_path)

    def _load_store(self, path: Path) -> Dict:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_store(self, path: Path, store: Dict):
        with open(path, "w") as f:
            json.dump(store, f, indent=2)

    def store_video(self, video_id: str, path: str):
        self.video_store[video_id] = path
        self._save_store(self.videos_path, self.video_store)

    def save_result(self, result_id: str, payload: Dict):
        self.results_store[result_id] = payload
        self._save_store(self.results_path, self.results_store)

    def get_video(self, video_id: str) -> str | None:
        return self.video_store.get(video_id)

    def get_result(self, result_id: str) -> Dict | None:
        return self.results_store.get(result_id)