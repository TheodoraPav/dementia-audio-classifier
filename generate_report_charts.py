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

print("\nColumns:", df.columns.tolist())
print("\nUnique Scenarios:", df['Scenario'].unique())
print("\nData shape:", df.shape)

# Helper function to extract run configuration from scenario
def get_run_config(scenario):
    """Extract configuration info (Baseline, Filtered, etc.)"""
    if 'Baseline' in scenario:
        return 'Baseline'
    elif 'Filtered' in scenario:
        return 'Filtered (Corr<0.95)'
    else:
        return 'Other'

# Improved comparison chart - top 3 runs with grouped bars
def create_grouped_comparison_chart(config1_label, config1_pattern, 
                                   config2_label, config2_pattern,
                                   title, output_file, metric='F1', top_n=3):

    # Get results for both configs
    config1_data = df[df['Scenario'].str.contains(config1_pattern, case=False, na=False)].copy()
    config2_data = df[df['Scenario'].str.contains(config2_pattern, case=False, na=False)].copy()
    
    if config1_data.empty or config2_data.empty:
        print(f"No data found for {title}")
        return
    
    # Add run config info
    config1_data['Run'] = config1_data['Scenario'].apply(get_run_config)
    config2_data['Run'] = config2_data['Scenario'].apply(get_run_config)
    
    # Get top N from each config
    top_config1 = config1_data.nlargest(top_n, metric)
    top_config2 = config2_data.nlargest(top_n, metric)
    
    # Create labels for each run (model + run type)
    top_config1['Label'] = top_config1.apply(
        lambda r: f"{r['Model'].replace('_', ' ')}\n({r['Run']})", axis=1
    )
    top_config2['Label'] = top_config2.apply(
        lambda r: f"{r['Model'].replace('_', ' ')}\n({r['Run']})", axis=1
    )
    
    # Take min length to ensure pairing
    n_runs = min(len(top_config1), len(top_config2), top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    color1 = '#2C3E50'
    color2 = '#E74C3C'
    
    x = np.arange(n_runs)
    width = 0.35
    
    # Get values
    vals1 = top_config1[metric].values[:n_runs]
    vals2 = top_config2[metric].values[:n_runs]
    labels = top_config1['Label'].values[:n_runs]
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, vals1, width, label=config1_label, 
                   color=color1, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, vals2, width, label=config2_label, 
                   color=color2, edgecolor='black', linewidth=1.2)
    
    # Find and highlight best overall
    max_val = max(vals1.max(), vals2.max())
    
    # Highlight best bar
    for bars in [bars1, bars2]:
        for bar in bars:
            if abs(bar.get_height() - max_val) < 0.001:
                bar.set_edgecolor('gold')
                bar.set_linewidth(3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add percentage difference between configs
    for i in range(n_runs):
        v1, v2 = vals1[i], vals2[i]
        diff = v2 - v1
        pct_diff = (diff / v1 * 100) if v1 != 0 else 0
        
        # Position annotation above both bars
        y_pos = max(v1, v2) + 0.04
        x_pos = x[i]
        
        # Neutral styling - no color coding
        symbol = '↑' if diff > 0 else '↓'
        
        ax.text(x_pos, y_pos, f'{symbol} {abs(pct_diff):.1f}%',
               ha='center', va='bottom', fontsize=9, 
               fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', linewidth=1.5, alpha=0.9))
    
    
    ax.set_xlabel('Top Performing Runs (Model + Configuration)', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max_val * 1.2)  # Extra space for annotations
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

# ==========================================
# 1. Audio Raw vs Text Raw
# ==========================================
print("\n1. Creating Audio Raw vs Text Raw comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Audio Only', 'AudioOnly_Raw',
        'Text Only', 'TextOnly_Raw',
        f'Audio Raw vs Text Raw ({metric})',
        os.path.join(OUTPUT_DIR, f'1_audio_vs_text_raw_{metric.lower()}.png'),
        metric=metric, top_n=3
    )

# ==========================================
# 2. Combined Raw (Early Fusion) vs Late Fusion
# ==========================================
print("\n2. Creating Early Fusion vs Late Fusion comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Early Fusion', 'Combined_Raw',
        'Late Fusion', 'Late_Fusion_Raw',
        f'Early Fusion vs Late Fusion ({metric})',
        os.path.join(OUTPUT_DIR, f'2_early_vs_late_fusion_{metric.lower()}.png'),
        metric=metric, top_n=3
    )

# ==========================================
# 3. Raw Audio vs Segmented Audio
# ==========================================
print("\n3. Creating Raw Audio vs Segmented Audio comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Raw Audio', 'AudioOnly_Raw',
        'Segmented Audio', 'AudioOnly_Segmented',
        f'Raw Audio vs Segmented Audio ({metric})',
        os.path.join(OUTPUT_DIR, f'3_raw_vs_segmented_audio_{metric.lower()}.png'),
        metric=metric, top_n=3
    )

# ==========================================
# 4. Raw Audio vs Audio Without Investigator (Continuous)
# ==========================================
print("\n4. Creating Raw Audio vs Continuous Audio comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Raw Audio', 'AudioOnly_Raw',
        'Continuous (No Inv.)', 'Continuous_Audio',
        f'Raw Audio vs Continuous Audio ({metric})',
        os.path.join(OUTPUT_DIR, f'4_raw_vs_continuous_audio_{metric.lower()}.png'),
        metric=metric, top_n=3
    )

# ==========================================
# 5. Text Raw vs Text Segmented
# ==========================================
print("\n5. Creating Text Raw vs Text Segmented comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Text Raw', 'TextOnly_Raw',
        'Text Segmented', 'TextOnly_Segmented',
        f'Text Raw vs Text Segmented ({metric})',
        os.path.join(OUTPUT_DIR, f'5_text_raw_vs_segmented_{metric.lower()}.png'),
        metric=metric, top_n=3
    )

# ==========================================
# 6. Early Fusion Raw vs Early Fusion Segmented
# ==========================================
print("\n6. Creating Early Fusion Raw vs Early Fusion Segmented comparison...")

for metric in ['F1', 'AUC']:
    create_grouped_comparison_chart(
        'Early Fusion Raw', 'Combined_Raw',
        'Early Fusion Segmented', 'Combined_Segmented',
        f'Early Fusion Raw vs Early Fusion Segmented ({metric})',
        os.path.join(OUTPUT_DIR, f'6_early_fusion_raw_vs_segmented_{metric.lower()}.png'),
        metric=metric, top_n=3
    )


# ==========================================
# 7. Effect of Correlation Filtering
# ==========================================
print("\n7. Creating Correlation Filtering Effect comparison...")

# For each model, compare Baseline vs Filtered
models = df['Model'].unique()
filter_data = []

for model in models:
    # Get baseline
    baseline = df[(df['Scenario'].str.contains('Combined_Raw_Baseline', case=False, na=False)) & 
                  (df['Model'] == model)]
    
    # Get filtered
    filtered = df[(df['Scenario'].str.contains('Combined_Raw_Filtered', case=False, na=False)) & 
                  (df['Model'] == model)]
    
    if not baseline.empty and not filtered.empty:
        for metric in ['F1', 'AUC']:
            baseline_val = baseline[metric].values[0]
            filtered_val = filtered[metric].values[0]
            delta = filtered_val - baseline_val
            
            filter_data.append({
                'Model': model,
                'Metric': metric,
                'Baseline': baseline_val,
                'Filtered': filtered_val,
                'Delta': delta
            })

if filter_data:
    filter_df = pd.DataFrame(filter_data)
    
    # Create plot for each metric
    for metric in ['F1', 'AUC']:
        metric_data = filter_df[filter_df['Metric'] == metric]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metric_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, metric_data['Baseline'], width, 
                       label='Baseline', color='#3498db')
        bars2 = ax.bar(x + width/2, metric_data['Filtered'], width, 
                       label='Filtered (Corr < 0.95)', color='#2ecc71')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Add delta annotations with percentage
        for i, (idx, row) in enumerate(metric_data.iterrows()):
            delta = row['Delta']
            baseline_val = row['Baseline']
            pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0
            
            # Position above the taller bar
            y_pos = max(row['Baseline'], row['Filtered']) + 0.02
            
            # Neutral styling - no color coding
            symbol = '↑' if delta > 0 else '↓'
            
            # Show percentage change
            ax.text(i, y_pos, f'{symbol} {abs(pct_change):.1f}%',
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', linewidth=1.5, alpha=0.9))
        
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Effect of Correlation Filtering on {metric}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_data['Model'])
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, f'7_correlation_filtering_{metric.lower()}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

print("\n" + "="*50)
print("All charts generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*50)
