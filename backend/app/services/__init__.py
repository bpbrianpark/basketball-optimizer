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
        """Statistics for training, returns feature array"""
        
        # Takes only the necessary columns
        columns = ['elbow_angle', 'shoulder_angle', 'wrist_angle']
        
        # Compute summary statistics - may change later
        stats = []
        for col in columns:
            if col in df.columns:
                mu = df[col].mean()
                sigma = df[col].std()
                
                if pd.isna(mu): 
                    mu = 0.0
                if pd.isna(sigma): 
                    sigma = 0.0
                
                stats.extend([mu, sigma])
            else:
                stats.extend([0.0, 0.0])
                
        features = stats # TODO: May add release_frame later (depends on how it is implemented elsewhere) 
        
        # numpy array
        feature_array = np.array(stats).reshape(1, -1)
    
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
    
    def text_feedback(self, prediction, feature_array):
        """Takes in dict and numpy feature array, returns text feedback based on prediction score"""
        
        score = prediction['score']
        strengths = []
        weaknesses = []
        
        
        feats = feature_array[0] 
        
        # Getting information out
        mean_elbow = feats[0]
        std_shoulder  = feats[3]
        std_wrist  = feats[5]
        
        # General Score Feedback
        if score >= 80:
            strengths.append("High consistency detected in your form.")
        elif score >= 50:
            strengths.append("There is room for improvement but getting there.")
        else:
            weaknesses.append("Overall mechanics inconsistent with training data.")
            
        # Elbow Feedback
        if 80 <= mean_elbow <= 100:
            strengths.append("Elbow angle is excellent.")
        elif mean_elbow > 110:
            weaknesses.append("Elbow is flaring out or too straight.")
        elif mean_elbow < 70:
            weaknesses.append("Elbow is bent too tight.")
            
        # Shoulder Feedback
        if std_shoulder > 15:
            weaknesses.append("Shoulder is unstable during the shot.")
        else:
            strengths.append("Upper arm stability is good.")
            
        # Wrist Feedback
        if std_wrist < 5: 
            weaknesses.append("Stiff wrist detected. Focus on flicking your wrist at release.")
        elif std_wrist > 15:
            strengths.append("Good follow-through and wrist flexion.")
            
        return strengths, weaknesses