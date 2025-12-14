import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Loads features.csv and labels.csv from data directory
def load_data():
    # Loading features.csv wtih error handling
    try:
        features_df = pd.read_csv('../data/raw/features.csv')
    except FileNotFoundError: 
        raise RuntimeError(f"features.csv was not found") 
    except Exception as e: 
        raise RuntimeError(f"Error in loading csv file: {e}")
    
    # Loading labels.csv with error handling
    try:
        labels_df = pd.read_csv('../data/raw/labels.csv')
    except FileNotFoundError: 
        raise RuntimeError(f"features.csv was not found") 
    except Exception as e: 
        raise RuntimeError(f"Error in loading csv file: {e}")
    
    # Merge dataframes on 'shot_id', deletes any rows where shot_id doesn't exist in both
    merged_df = pd.merge(features_df, labels_df, on='shot_id', how='inner')
    
    # Saves merged_df
    merged_df.to_csv('../data/processed/merged_data.csv', index=False)
    
    return merged_df

# Splitting data
def split(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # 80/20 split
    return train_df, test_df

# Loads rf model - default parameters
def instantiate_model():
    model = RandomForestClassifier()
    return model

# Training model on train_df; target variable is 'is_goal' (MAY CHANGE LATER)
def train_model(model, train_df):
    # Split into X_train and y_train
    X_train = train_df.drop(columns=['shot_id', 'shot_made'])
    y_train = train_df['shot_made']
    
    print("Starting model training")
    model.fit(X_train, y_train)
    print("Model training complete")
    
    return model

def evaluation(model, test_df):
    # Split into X_test and y_test
    X_test = test_df.drop(columns=['shot_id', 'shot_made'])
    y_test = test_df['shot_made']
    
    # predictions + probailities of predictions for positive class
    predictions = model.predict(X_test)
    predict_probabililties = model.predict_proba(X_test)
    
    # Get positive and negative probabilities - may be useful later
    pos_prob = predict_probabililties[:, 1]
    neg_prob = predict_probabililties[:, 0]
    
    # Ensure correct shape of predictions and predict_probabilities
    assert len(predictions) == len(y_test)
    assert predict_probabililties[0] == len(y_test)
    
    return predictions, predict_probabililties

def evaluate_metrics(model, test_df):
    # TODO - get validation accuracy/f1/cm
    
def save_model(model):
    # TODO - save model and json with hyperparameters/information about model