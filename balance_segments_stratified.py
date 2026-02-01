import pandas as pd
import numpy as np

def balance_segments_stratified(input_file, output_file, target_per_patient='median'):
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # Extract patient ID
    df['patient_id'] = df['file_name'].apply(lambda x: x.split('_')[0])
    
    # Calculate targe
    segments_per_patient = df.groupby('patient_id').size()
    
    if target_per_patient == 'median':
        target = int(segments_per_patient.median())
    elif target_per_patient == 'mean':
        target = int(segments_per_patient.mean())
    else:
        target = int(target_per_patient)
    
    print(f"\nTarget segments per patient: {target}")
    print(f"  Min in dataset: {segments_per_patient.min()}")
    print(f"  Median: {segments_per_patient.median():.1f}")
    print(f"  Max in dataset: {segments_per_patient.max()}")
    
    # Stratified sampling
    balanced_dfs = []
    
    for patient_id, group in df.groupby('patient_id'):
        n_segments = len(group)
        
        if n_segments <= target:
            # Keep all if below target
            balanced_dfs.append(group)
        else:
            # Sample down to target
            sampled_group = group.sample(n=target, random_state=42)
            print(f"  {patient_id}: {n_segments} -> {target} segments")
            balanced_dfs.append(sampled_group)
    
    # Combine all patients
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Remove temporary patient_id column
    balanced_df = balanced_df.drop(columns=['patient_id'])
    
    print(f"\nBalanced dataset: {len(balanced_df)} samples")
    print(f"Removed: {len(df) - len(balanced_df)} segments")
    
    # Show class distribution
    print(f"\nClass distribution:")
    print(balanced_df['label'].value_counts())
    
    # Save balanced dataset
    balanced_df.to_excel(output_file, index=False)
    print(f"\nSaved balanced dataset to: {output_file}")
    
    return balanced_df

if __name__ == "__main__":
    import os
    
    # Try different strategies
    input_file = 'data/processed/features_segmented.xlsx'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
    else:
        # Strategy 1: Use median (more robust to outliers)
        print("="*70)
        print("TESTING MEDIAN-BASED BALANCING")
        print("="*70)
        balance_segments_stratified(input_file, 'data/processed/features_segmented_balanced_median.xlsx', 'median')
        
        print("\n" + "="*70)
        print("TESTING MEAN-BASED BALANCING")
        print("="*70)
        balance_segments_stratified(input_file, 'data/processed/features_segmented_balanced_mean.xlsx', 'mean')
