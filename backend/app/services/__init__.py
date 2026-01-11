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
    
    def feature_extraction(self, df):
        # SUMMARY STATISTICS NEED TO BE DETERMINED ELSEWHERE FIRST
        """Statistics for training, returns feature array"""
        
        # Takes only the necessary columns
        columns = ['elbow_angle', 'shoulder_angle', 'wrist_angle']
        filtered_data = df[columns]
        b 
        # Flattern to numpy array
        feature_array = filtered_data.values.flatten().reshape(1, -1)
    
        return feature_array
        
    def predict(self, features):
        """Generates prediction/score"""
        if self.classifier is None:
            raise ValueError("Model is not loaded. Cannot make predictions.")
        
        # Predict probabilities (two classes: good/bad)
        all_probs = self.classifier.predict_proba(features)[0]
        
        # Find probability of good score, assuming class '1' is good and scale it up by 100
        prob_good = all_probs[1] 
        score = int(prob_good * 100)

        # Find which index is higher and get the label out of it
        predicted_index = all_probs.argmax()
        predicted_label = self.classifier.classes_[predicted_index]

        # Dictionary output
        return {
            "score": score,
            "probability_good": prob_good,
            "predicted_label": predicted_label
        }
    