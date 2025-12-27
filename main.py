import os
import sys
import pandas as pd

# Ensure src module is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extraction import extract_features
from src.preprocess import remove_correlated_features

def main():
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "Full_wave_enhanced_audio")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "outputs")
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("========================================")
    print("   Medical Audio Classification Pipeline")
    print("========================================")

    # 2. Extract Features (Base)
    print(f"\n[Step 1] Ensuring Features Exist...")
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at: {RAW_DATA_DIR}")
        return

    base_features_file = extract_features(RAW_DATA_DIR, FEATURES_DIR)
    if not base_features_file:
        return
    
    # 3. Define Experiments
    # Simple configuration to control the pipeline
    EXPERIMENTS = [
        {
            "name": "Baseline", 
            "use_filter": False, 
            "suffix": "original"
        },
        {
            "name": "Filtered (Corr < 0.95)", 
            "use_filter": True, 
            "suffix": "corr95"
        }
    ]


    # 4. Run Experiments Loop
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        current_file = base_features_file
        
        # Step A: Correlation Filter
        if exp["use_filter"]:
            output_path = os.path.join(FEATURES_DIR, f"features_{exp['suffix']}_filtered.xlsx")
            print("  -> Applying Correlation Filter...")
            current_file = remove_correlated_features(current_file, output_path, threshold=0.95)
            if not current_file: 
                print("Skipping...")
                continue

if __name__ == "__main__":
    main()
