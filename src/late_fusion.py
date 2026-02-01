import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

def fuse_predictions(file_path_1, file_path_2, output_dir, weights=(0.5, 0.5)):
    """
    Fuses predictions from two CSV files using weighted soft voting.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # Merge on file_name
    merged = pd.merge(df1, df2, on='file_name', suffixes=('_1', '_2'), how='inner')
    
    if merged.empty:
        print("Error: No matching files found for fusion.")
        return None

    # Get True Labels (assume they are consistent)
    y_true = merged['y_true_1']
    
    # Get Probabilities (assume column names are 'y_probs')
    prob1 = merged['y_probs_1']
    prob2 = merged['y_probs_2']

    # Soft Voting
    w1, w2 = weights
    fused_probs = (w1 * prob1) + (w2 * prob2)
    fused_preds = (fused_probs >= 0.5).astype(int)

    # Calculate Metrics
    acc = accuracy_score(y_true, fused_preds)
    f1 = f1_score(y_true, fused_preds)
    precision = precision_score(y_true, fused_preds, zero_division=0)
    recall = recall_score(y_true, fused_preds, zero_division=0)

    try:
        auc = roc_auc_score(y_true, fused_probs)
    except:
        auc = 0.0
    
    metrics = {
        "Accuracy": acc,
        "F1": f1,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "Classification Error": 1 - acc
    }

    # Save Fused Predictions
    merged['fused_probs'] = fused_probs
    merged['fused_preds'] = fused_preds
    merged.to_csv(os.path.join(output_dir, "fused_predictions.csv"), index=False)

    return metrics
