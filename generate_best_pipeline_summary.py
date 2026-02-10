import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "outputs", "final_consolidated_report.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reported")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load results
print("Loading results from:", RESULTS_FILE)
df = pd.read_excel(RESULTS_FILE)

# Extract pipeline name from scenario
def get_pipeline_name(scenario):
    # Remove the experiment suffix (Baseline/Filtered)
    if '_Baseline' in scenario:
        return scenario.split('_Baseline')[0]
    elif '_Filtered' in scenario:
        return scenario.split('_Filtered')[0]
    return scenario

# Extract run configuration
def get_run_config(scenario):
    if 'Baseline' in scenario:
        return 'Baseline'
    elif 'Filtered' in scenario:
        return 'Filtered (Corr<0.95)'
    else:
        return 'Other'

# Add pipeline and run info
df['Pipeline'] = df['Scenario'].apply(get_pipeline_name)
df['Run'] = df['Scenario'].apply(get_run_config)

# Define all pipelines we want to compare
pipelines = [
    'AudioOnly_Raw',
    'AudioOnly_Segmented', 
    'TextOnly_Raw',
    'TextOnly_Segmented',
    'Combined_Raw',
    'Combined_Segmented',
    'Late_Fusion_Raw',
    'Continuous_Audio'
]

# Create final summary chart
def create_best_pipeline_chart(metric='F1'):
    best_results = []
    
    for pipeline in pipelines:
        # Get all runs for this pipeline
        pipeline_data = df[df['Pipeline'] == pipeline]
        
        if pipeline_data.empty:
            continue
        
        # Find the best run
        best_run = pipeline_data.nlargest(1, metric).iloc[0]
        
        # Create readable label
        label = f"{pipeline.replace('_', ' ')}\n{best_run['Model'].replace('_', ' ')} ({best_run['Run']})"
        
        best_results.append({
            'Pipeline': pipeline,
            'Label': label,
            'Model': best_run['Model'],
            'Run': best_run['Run'],
            'Scenario': best_run['Scenario'],
            metric: best_run[metric]
        })
    
    if not best_results:
        print(f"No data found for best pipeline chart ({metric})")
        return
    
    results_df = pd.DataFrame(best_results)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Professional color palette - different color per pipeline type
    colors = {
        'AudioOnly_Raw': '#34495E',
        'AudioOnly_Segmented': '#2C3E50',
        'TextOnly_Raw': '#E74C3C',
        'TextOnly_Segmented': '#C0392B',
        'Combined_Raw': '#16A085',
        'Combined_Segmented': '#148F77',
        'Late_Fusion_Raw': '#8E44AD',
        'Continuous_Audio': '#2980B9'
    }
    
    bar_colors = [colors.get(p, '#95A5A6') for p in results_df['Pipeline']]
    
    x = np.arange(len(results_df))
    bars = ax.bar(x, results_df[metric], color=bar_colors, 
                  edgecolor='black', linewidth=1.5)
    
    # Highlight best overall
    max_val = results_df[metric].max()
    for bar, val in zip(bars, results_df[metric]):
        if abs(val - max_val) < 0.001:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, results_df[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Pipeline (Best Model + Configuration)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Best Performance per Pipeline ({metric})', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Label'], fontsize=8, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'0_best_per_pipeline_{metric.lower()}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Print summary
    print(f"\nBest runs per pipeline ({metric}):")
    for _, row in results_df.iterrows():
        print(f"  {row['Pipeline']:25s} -> {row['Model']:15s} ({row['Run']:20s}): {row[metric]:.3f}")

# Generate charts for both metrics
print("\n" + "="*60)
print("Creating Best-per-Pipeline Summary Charts")
print("="*60)

for metric in ['F1', 'AUC']:
    print(f"\nGenerating chart for {metric}...")
    create_best_pipeline_chart(metric=metric)

print("\n" + "="*60)
print("Summary charts generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)
