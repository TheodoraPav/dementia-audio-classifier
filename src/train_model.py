import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import model definitions
from src.models import get_models

def train_and_evaluate(data_file, output_dir, scenario_name="Baseline"):
    print(f"\n--- Starting Model Training: {scenario_name} (LOOCV) ---")
    
    # Load Data
    try:
        df = pd.read_excel(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return []

    target_col = 'label'
    if target_col not in df.columns:
        print("Error: 'label' column missing.")
        return []

    X = df.drop(columns=[target_col, 'file_name'], errors='ignore') 
    y = df[target_col]

    # Define Models
    models = get_models()

    cv = LeaveOneOut()
    roc_data_list = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        start_time = time.time()
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        try:
            # Cross-Validated Predictions
            y_probs = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]
            y_pred = (y_probs >= 0.5).astype(int)
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Calculate Metrics
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_probs)

            print(f"Results for {name}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}\n")

            # Store data for plotting/results
            roc_data_list.append({
                'name': name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc
            })
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")



if __name__ == "__main__":
    # Test run
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = os.path.join(BASE_DIR, "features", "features_dataset.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    train_and_evaluate(DATA_FILE, OUTPUT_DIR)
