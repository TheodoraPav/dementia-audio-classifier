import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import metrics utility
from src.utils import evaluate_predictions, save_metrics_to_excel, plot_roc_comparison
# Import model definitions
from src.models import get_models

def train_and_evaluate(data_file, output_dir, scenario_name="Baseline", validation_method="group_kfold"):
    print(f"\n--- Starting Training: {scenario_name} [{validation_method}] ---")

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


    groups = None
    cv = None

    # Setup Validation Strategy
    if validation_method == "group_kfold":
        if 'file_name' not in df.columns:
             print("Error: 'file_name' missing for GroupKFold.")
             return []
             
        # Extract Groups (Subject IDs)
        def extract_id(filename):
            name = os.path.splitext(filename)[0]
            parts = name.rsplit('_', 1) 
            if len(parts) > 1: return parts[0]
            return name

        groups = df['file_name'].apply(extract_id).values
        print(f"  -> Found {len(np.unique(groups))} unique subjects (Groups).")
        cv = GroupKFold(n_splits=5)
        
    elif validation_method == "loocv":
        print(f"  -> Using Leave-One-Out Cross-Validation on {len(df)} samples.")
        cv = LeaveOneOut()
        groups = None
        
    else:
        print(f"Error: Unknown validation method {validation_method}")
        return []

    # Define Models
    models = get_models()
    
    metrics_list = []
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
            y_probs = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, method='predict_proba')[:, 1]
            y_pred = (y_probs >= 0.5).astype(int)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Calculate Metrics using util
            metrics = evaluate_predictions(y, y_pred, y_probs, model_name=name)
            metrics['Scenario'] = scenario_name
            metrics['Runtime (s)'] = round(elapsed_time, 4) # Add Runtime

            metrics_list.append(metrics)

            # Store data for plotting
            roc_data_list.append({
                'name': name,
                'y_true': y,
                'y_probs': y_probs,
                'auc': metrics.get('AUC', 0)
            })

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Save Results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate unique filename safe for the scenario
    safe_scenario = "".join([c if c.isalnum() else "_" for c in scenario_name]) + "_" + validation_method

    # Save Metrics Table
    save_metrics_to_excel(metrics_list, os.path.join(output_dir, f"results_{safe_scenario}.xlsx"))

    # Save ROC Plot
    plot_roc_comparison(roc_data_list, os.path.join(output_dir, f"roc_{safe_scenario}.png"))

    return metrics_list

if __name__ == "__main__":
    # Test run
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = os.path.join(BASE_DIR, "features", "features_dataset.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    train_and_evaluate(DATA_FILE, OUTPUT_DIR)