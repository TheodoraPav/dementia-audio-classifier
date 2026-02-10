import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import seaborn as sns

def evaluate_predictions(y_true, y_pred, y_probs=None, model_name="Model"):
    """
    Calculates performance metrics for a classification model.
    Uses macro averaging for Precision, Recall, and F1 to give equal weight to both classes.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Classification Error": 1 - accuracy_score(y_true, y_pred)
    }

    if y_probs is not None:
        try:
            if len(np.unique(y_true)) < 2:
                print(f"Warning: {model_name} - Only one class present in y_true, cannot calculate AUC")
                metrics["AUC"] = np.nan
            elif len(np.unique(y_probs)) < 2:
                print(f"Warning: {model_name} - All predicted probabilities are identical, cannot calculate AUC")
                metrics["AUC"] = np.nan
            else:
                metrics["AUC"] = roc_auc_score(y_true, y_probs)
        except (ValueError, Exception) as e:
            print(f"Warning: {model_name} - Error calculating AUC: {e}")
            metrics["AUC"] = np.nan

    return metrics


def save_feature_importance(model, feature_names, output_image='feature_importance.png', output_csv='feature_weights.csv'):
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        coefs = model.feature_importances_
    else:
        print("Model does not have coef_ or feature_importances_")
        return

    transcript_features = [
        'filler_ratio', 'pause_ratio', 'rep_ratio', 
        'correction_ratio', 'error_ratio', 'self_correction_ratio', 
        'words_per_minute'
    ]
    
    def get_category(feat_name):
        return "Transcript" if feat_name in transcript_features else "Audio"

    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefs
    })

    features_df['Category'] = features_df['Feature'].apply(get_category)
    features_df['AbsWeight'] = features_df['Weight'].abs()
    features_df = features_df.sort_values(by='AbsWeight', ascending=False)
    features_df.to_csv(output_csv, index=False)

    top_df = features_df.head(20).copy()
    top_df = top_df.sort_values(by='Weight', ascending=True)

    plt.figure(figsize=(12, 10))
    palette = {"Audio": "#1f77b4", "Transcript": "#ff7f0e"}
    
    sns.barplot(data=top_df, x='Weight', y='Feature', hue='Category', dodge=False, palette=palette)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Labeling and Formatting
    plt.xlabel('SVM Coefficient Weight (Magnitude & Direction)', fontsize=12)
    plt.title('Top 20 Features Influencing Dementia Detection', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(title='Feature Type')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Feature importance plot saved to: {output_image}")