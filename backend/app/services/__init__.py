import numpy as np
import pandas as pd

import joblib
import logging

class ModelService:
    def __init__(self, model_path: str):
        """Initializaing model service by loading the model from the specified path"""
        self.classifier = None # Set to none initially
        try:
            self.classifier = joblib.load(model_path) # Load model if we are able too
        except Exception as e:
            # Warnings if no path found, but service can still run since classifier is None
            logging.warning("Model file not found. Service running without predictions.")
            print(f"Warning: Model file not found. {e}")

    def has_model(self):
        """Check if the model is loaded successfully"""
        return self.classifier is not None
    
    