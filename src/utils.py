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
        "F1": f1_score(y_true, y_pred, zero_division=0),
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
    Produces Grouped Bar Charts for Acc, F1, Precision, Recall, Classification Error
    comparing Segmented vs. Raw audio,
    and a standalone AUC Bar Chart.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_df = df.copy()
    
    def parse_scenario(s):
        # 1. Determine Audio Type (independent of naming)
        # Late Fusion is based on predictions from Raw models, so we categorize it as Raw.
        if s.startswith("Late_Fusion"):
             audio_type = "Raw"
        elif "Segmented" in s:
             audio_type = "Segmented"
        else:
             audio_type = "Raw"

        # 2. explicit Renames to friendly names
        # Combined -> Early Fusion
        # Late_Fusion -> Late Fusion
        config = s.replace("Combined", "Early Fusion").replace("Late_Fusion", "Late Fusion")
        
        # 3. Remove Type Tags (Segmented/Raw) to allow grouping on X-axis
        tags = ["_Segmented_", "Segmented_", "_Segmented", 
                "_Raw_", "Raw_", "_Raw"]
        
        for tag in tags:
            config = config.replace(tag, " ")
            
        # 4. Cleanup
        config = config.replace("_", " ").strip()
        # Collapse multiple spaces
        config = " ".join(config.split())
        
        return audio_type, config

    plot_df[['Audio Type', 'Configuration']] = plot_df['Scenario'].apply(
        lambda x: pd.Series(parse_scenario(x))
    )

    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'Classification Error']
    
    # 3x2 Grid for 5 metrics
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric in plot_df.columns:
            sns.barplot(data=plot_df, x='Configuration', y=metric, hue='Audio Type', 
                        ax=axes[i], palette="viridis", errorbar=None)

            # Add value annotations
            for container in axes[i].containers:
                axes[i].bar_label(container, fmt='%.2f', padding=3, fontsize=9)

            axes[i].set_title(f'Comparative {metric}', fontsize=14, fontweight='bold', pad=10)
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Configuration')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=15, ha='right')
            
            if i == 0:
                axes[i].legend(title='Audio Type', loc='upper left')
            else:
                if axes[i].get_legend(): axes[i].get_legend().remove()
        else:
            axes[i].set_visible(False)

    # Hide empty 6th subplot
    if len(metrics) < len(axes):
        for j in range(len(metrics), len(axes)):
            fig.delaxes(axes[j])

    plt.suptitle('Global Pipeline Performance', fontsize=22, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_metrics_bar_charts.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # AUC Bar Chart
    plt.figure(figsize=(12, 7))
    if 'AUC' in plot_df.columns:
        ax = sns.barplot(data=plot_df, x='Configuration', y='AUC', hue='Audio Type', palette="magma", errorbar=None)
        
        # Add value annotations for AUC
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

        plt.title('Consolidated AUC Performance', fontsize=16, pad=20)
        plt.ylabel('AUC Score')
        plt.xlabel('Configuration')
        plt.xticks(rotation=15, ha='right')
        plt.legend(title='Audio Type', loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'auc_comparison.png'), dpi=300)
        plt.close()

    print(f"Visualizations generated in {output_dir}")

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