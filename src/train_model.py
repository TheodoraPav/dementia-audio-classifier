import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import evaluate_predictions, save_metrics_to_excel, plot_roc_comparison, save_feature_importance
from src.models import get_models
from src.preprocess import get_uncorrelated_features
from src.feature_selection import get_selected_features

def train_and_evaluate(data_file, output_dir, scenario_name="Baseline", validation_method="group_kfold", use_filter=False, use_selection=False):
    print(f"\nStarting Training: {scenario_name} [{validation_method}]")

    #1. Load Data
    try:
        df = pd.read_excel(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return []

    target_col = 'label'
    if target_col not in df.columns:
        print("Error: 'label' column missing.")
        return []

    remove_chroma = True

    if remove_chroma:
        df = df.drop(columns=[c for c in df.columns if 'chroma' in c.lower()])
        print("Running without Chroma features...")


    X = df.drop(columns=[target_col, 'file_name'], errors='ignore')
    y = df[target_col]


    groups = None
    cv = None

    #2. Setup Validation Strategy
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
        cv = GroupKFold(n_splits=5)
        
    elif validation_method == "loocv":
        cv = LeaveOneOut()
        groups = None
        
    else:
        print(f"Error: Unknown validation method {validation_method}")
        return []

    #3. Define Models
    models = get_models()
    
    metrics_list = []
    roc_data_list = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        y_true_all = []
        y_probs_all = []
        start_time = time.time()
        
        try:
            #4. Manual Cross-Validation Loop
            for train_idx, val_idx in cv.split(X, y, groups=groups):
                
                #4.1 Split Data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                #4.2 Correlation Filter
                if use_filter:
                    keep_cols, _ = get_uncorrelated_features(X_train, threshold=0.95)
                    X_train = X_train[keep_cols]
                    X_val = X_val[keep_cols]
                
                #4.3 Feature Selection
                if use_selection:
                    selected_feats = get_selected_features(X_train, y_train, top_n=30)
                    if not selected_feats:
                         pass
                    else:
                        X_train = X_train[selected_feats]
                        X_val = X_val[selected_feats]

                #4.4 Pipeline (Scaler + Model)
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                #4.5 Train
                pipeline.fit(X_train, y_train)
                
                #4.6 Predict
                probs = pipeline.predict_proba(X_val)[:, 1]
                
                y_true_all.extend(y_val)
                y_probs_all.extend(probs)
            
            #5. Aggregate Results
            y_true_all = np.array(y_true_all)
            y_probs_all = np.array(y_probs_all)
            y_pred_all = (y_probs_all >= 0.5).astype(int)
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            #6. Calculate Metrics using util
            metrics = evaluate_predictions(y_true_all, y_pred_all, y_probs_all, model_name=name)
            metrics['Scenario'] = scenario_name
            metrics['Runtime (s)'] = round(elapsed_time, 4) # Add Runtime

            metrics_list.append(metrics)

            #7. Store data for plotting
            roc_data_list.append({
                'name': name,
                'y_true': y_true_all,
                'y_probs': y_probs_all,
                'auc': metrics.get('AUC', 0)
            })

            if name == "SVM_Linear":
                save_feature_importance(model, X_train.columns, output_image=os.path.join(output_dir, scenario_name + '_final_svm_features.png'), output_csv=os.path.join(output_dir, scenario_name +'_final_svm_weights.csv'))

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

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