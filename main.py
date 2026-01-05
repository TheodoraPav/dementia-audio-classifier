import os
import sys
import pandas as pd

# Ensure src module is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extraction import extract_features
from src.preprocess import remove_correlated_features
from src.feature_selection import select_common_features
from src.train_model import train_and_evaluate
from src.preprocess_segments import preprocess_dataset
from src.utils import generate_global_performance_charts

def main():
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "Full_wave_enhanced_audio")
    SEGMENTED_DATA_DIR = os.path.join(BASE_DIR, "data", "segmented_audio")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "outputs")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("========================================")
    print("   Medical Audio Classification Pipeline")
    print("========================================")

    # [Step 0] Check for Segmented Data
    print(f"\n[Step 0] Checking for Segmented Data")
    run_preprocessing = False

    if not os.path.exists(SEGMENTED_DATA_DIR) or not os.listdir(SEGMENTED_DATA_DIR):
        run_preprocessing = True
    else:
        print(f"Segmented data found at: {SEGMENTED_DATA_DIR}")
        user_input = input("Do you want to overwrite existing segments? (y/n): ").strip().lower()
        
        if user_input in ['y', 'yes']:
            print("Overwrite selected.")
            run_preprocessing = True
        else:
            print("Skipping preprocessing. Using existing data.")

    if run_preprocessing:
        preprocess_dataset(RAW_DATA_DIR, SEGMENTED_DATA_DIR, 5.0)

    # 2. Extract Features (Base)
    print(f"\n[Step 1] Ensuring Features Exist...")
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at: {RAW_DATA_DIR}")
        return

    features_file_path = os.path.join(FEATURES_DIR, "features_dataset.xlsx")
    
    # Check if features already exist
    should_extract = True
    if os.path.exists(features_file_path):
        print(f"Features file found at: {features_file_path}")
        user_response = input("Do you want to overwrite it and re-extract features? (y/n): ").strip().lower()
        if user_response != 'y':
            should_extract = False
            print("Using existing features.")
        else:
            print("Re-extracting features...")
    
    if should_extract:
        base_features_file = extract_features(RAW_DATA_DIR, FEATURES_DIR)
        if not base_features_file:
            return
    else:
        base_features_file = features_file_path
    
    # 3. Define Experiments
    # Simple configuration to control the pipeline
    EXPERIMENTS = [
        {
            "name": "Baseline",
            "use_filter": False,
            "use_selection": False,
            "suffix": "original"
        },
        {
            "name": "Filtered (Corr < 0.95)",
            "use_filter": True,
            "use_selection": False,
            "suffix": "corr95"
        },
        {
            "name": "Common Features (Intersection)",
            "use_filter": False,
            "use_selection": True,
            "suffix": "intersection"
        },
        {
            "name": "Combined (Filter + Selection)",
            "use_filter": True,
            "use_selection": True,
            "suffix": "combined"
        }
    ]

    all_metrics = []

    # ==========================================
    # PIPELINE A: Segmented + GroupKFold
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE A: Segmented Audio + GroupKFold")
    print("="*50)
    
    feats_segmented = os.path.join(FEATURES_DIR, "features_segmented.xlsx")
    
    # Extract Features A
    if not os.path.exists(feats_segmented):
         extract_features(SEGMENTED_DATA_DIR, FEATURES_DIR, "features_segmented.xlsx")
    else:
         print(f"Features already exist at {feats_segmented}")
         # Ask user if they want to recreate
         user_input = input("Do you want to overwrite and re-extract segmented features? (y/n): ").strip().lower()
         if user_input in ['y', 'yes']:
             print("Re-extracting features...")
             extract_features(SEGMENTED_DATA_DIR, FEATURES_DIR, "features_segmented.xlsx")
         else:
             print("Using existing segmented features.")
    
    for exp in EXPERIMENTS:
        current_file = feats_segmented
        suffix = exp["suffix"]
        
        # Apply Filters/Selection
        if exp["use_filter"]:
            out = os.path.join(FEATURES_DIR, f"features_segmented_{suffix}.xlsx")
            if not os.path.exists(out):
                 current_file = remove_correlated_features(current_file, out, threshold=0.95)
            else:
                 current_file = out
                 
        if exp["use_selection"]:
             out_sel = os.path.join(FEATURES_DIR, f"features_segmented_{exp['suffix']}_ready.xlsx")
             current_file = select_common_features(current_file, out_sel, top_n=20)
        
        if not current_file: continue
        
        # RUN TRAINING A (Default is GroupKFold in train_model.py)
        metrics = train_and_evaluate(
            current_file, 
            RESULTS_DIR, 
            scenario_name=f"Prop_{exp['name']}", 
            validation_method="group_kfold"
        )
        all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE B: Baseline (Raw Audio + LOOCV)
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE B: Raw Audio + LOOCV")
    print("="*50)

    # We use the raw-extracted features here
    current_base = base_features_file # This was extracted in 'Step 1' block above

    # 4. Run Experiments Loop
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n--- Running Experiment: {exp_name} ---")

        current_file = current_base

        # Step A: Correlation Filter
        if exp["use_filter"]:
            output_path = os.path.join(FEATURES_DIR, f"features_{exp['suffix']}_filtered.xlsx")
            print("  -> Applying Correlation Filter...")
            current_file = remove_correlated_features(current_file, output_path, threshold=0.95)
            if not current_file:
                print("Skipping...")
                continue

        # Step B: Feature Selection (Intersection)
        if exp["use_selection"]:
            output_path = os.path.join(FEATURES_DIR, f"features_{exp['suffix']}_selected.xlsx")
            print("  -> Applying Feature Selection (Intersection)...")
            current_file = select_common_features(current_file, output_path, top_n=20)
            if not current_file:
                print("Skipping...")
                continue

        # Step C: Train
        metrics = train_and_evaluate(
            current_file, 
            RESULTS_DIR, 
            scenario_name=f"Base_{exp['name']}",
            validation_method="loocv"
        )
        all_metrics.extend(metrics)

    # 5. Consolidate Results
    print("\n========================================")
    print("Consolidated Results")
    print("========================================")

    if all_metrics:
        final_df = pd.DataFrame(all_metrics)
        # Reorder columns
        cols = ['Scenario', 'Model'] + [c for c in final_df.columns if c not in ['Scenario', 'Model']]
        final_df = final_df[cols]

        output_file = os.path.join(RESULTS_DIR, "final_consolidated_report.xlsx")
        final_df.to_excel(output_file, index=False)

        print(final_df.to_string(index=False))
        print(f"\nFull report saved to: {output_file}")
        
        # Sort by Scenario name for better plotting
        final_df = final_df.sort_values('Scenario')
        generate_global_performance_charts(final_df, RESULTS_DIR)
    else:
        print("No metrics collected.")

if __name__ == "__main__":
    main()