import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import seaborn as sns

def evaluate_predictions(y_true, y_pred, y_probs=None, model_name="Model"):
    """
    Calculates performance metrics for a classification model.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_probs is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["AUC"] = np.nan

    return metrics

def plot_roc_comparison(results_list, output_file):
    """
    Plots ROC curves for multiple models.
    results_list: List of dicts, each containing:
                  {'name': str, 'y_true': array, 'y_probs': array, 'auc': float}
    """
    plt.figure(figsize=(10, 8))

    for res in results_list:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
        plt.plot(fpr, tpr, label=f"{res['name']} (AUC = {res['auc']:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")

    plt.savefig(output_file)
    print(f"ROC Curve saved to {output_file}")
    plt.close()

def save_metrics_to_excel(metrics_list, output_file):
    """
    Saves a list of metric dictionaries to an Excel file.
    """
    df = pd.DataFrame(metrics_list)
    if not df.empty and 'AUC' in df.columns:
        df = df.sort_values(by='AUC', ascending=False)

    df.to_excel(output_file, index=False)
    print(f"Metrics saved to {output_file}")
    print("\n=== Model Performance ===")
    print(df.to_string(index=False))



def generate_global_performance_charts(df, output_dir):
    """
    Produces a 2x2 Faceted Heatmap for Acc, F1, Precision, Recall
    and a standalone AUC Heatmap.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- PART 1: THE 4-METRIC DASHBOARD ---
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
    palettes = ['YlGnBu', 'YlOrRd', 'BuPu', 'GnBu']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        pivot = df.pivot_table(index='Scenario', columns='Model', values=metric, observed=False)

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=palettes[i],
                    ax=axes[i], cbar_kws={'label': metric})

        axes[i].set_title(f'Comparative {metric}', fontsize=14, fontweight='bold', pad=10)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        # Tilt x-labels for better readability
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=15, ha='right')

    plt.suptitle('Global Pipeline Performance: Multi-Metric Analysis', fontsize=22, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_metrics_dashboard.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- PART 2: THE AUC HEATMAP ---
    plt.figure(figsize=(10, 7))
    pivot_auc = df.pivot_table(index='Scenario', columns='Model', values='AUC', observed=False)

    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)

    plt.title('Consolidated AUC Performance Map', fontsize=16, pad=20)
    plt.ylabel('Experimental Scenario', fontsize=12)
    plt.xlabel('Classifier Architecture', fontsize=12)
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_heatmap.png'), dpi=300)
    plt.close()

    print(f"Visualizations generated in {output_dir}")

def save_feature_importance(model, feature_names, output_image='feature_importance.png', output_csv='feature_weights.csv'):
    # 1. Extract Coefficients
    coefs = model.coef_[0]

    # 2. Create a DataFrame
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefs
    })

    # Add absolute values to identify the most influential features regardless of direction
    features_df['AbsWeight'] = features_df['Weight'].abs()

    # Sort features from most impactful to least impactful
    features_df = features_df.sort_values(by='AbsWeight', ascending=False)

    # 3. Save numerical weights to a CSV file
    features_df.to_csv(output_csv, index=False)
    print(f"Numerical weights saved to: {output_csv}")

    # 4. Prepare Visualization
    top_df = features_df.head(20).sort_values(by='Weight')

    plt.figure(figsize=(12, 10))

    # Conditional Coloring:
    # Red (#d63031) for positive weights (indicates Dementia-prone traits)
    # Blue (#0984e3) for negative weights (indicates Healthy-control traits)
    colors = ['#d63031' if x > 0 else '#0984e3' for x in top_df['Weight']]

    bars = plt.barh(top_df['Feature'], top_df['Weight'], color=colors, alpha=0.8)

    # Add a vertical line at zero to clearly separate positive and negative influences
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Labeling and Formatting
    plt.xlabel('SVM Coefficient Weight (Magnitude & Direction)', fontsize=12)
    plt.title('Top 20 Features Influencing Dementia Detection\n(Red: Dementia-prone | Blue: Healthy-prone)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()

    print(f"Feature importance plot saved to: {output_image}")