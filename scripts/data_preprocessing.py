import pandas as pd
from sklearn.model_selection import train_test_split

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