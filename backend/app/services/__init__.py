import joblib
import logging

class ModelService:
    def __init__(self, model_path: str):
        self.classifier = None
        try:
            self.classifier = joblib.load(model_path)
        except Exception as e:
            logging.warning("Model file not found. Service running without predictions.")
            print(f"Warning: Model file not found. {e}")

    def has_model(self):
        return self.classifier is not None