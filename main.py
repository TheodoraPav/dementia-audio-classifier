import os
import sys
import pandas as pd

# Ensure src module is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extraction import extract_features
from src.train_model import train_and_evaluate
from src.preprocess_segments import preprocess_with_pyannote, preprocess_with_transcript
from src.utils import generate_global_performance_charts
from src.merge_txt_feature import extract_text_features_from_dir

def main():
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "Full_wave_enhanced_audio")
    SEGMENTED_DATA_DIR = os.path.join(BASE_DIR, "data", "segmented_audio")
    FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "outputs")
    TRANSCRIPTION_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "transcription")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("========================================")
    print("   Medical Audio Classification Pipeline")
    print("========================================")

    # 1. Check for Segmented Data
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
        print("Select Preprocessing Method:")
        print("1. Pyannote (Speaker Diarization)")
        print("2. Transcript-Based (Exact Timestamps)")
        method = input("Enter method (1-2) [Default: 2]: ").strip()
        
        if method == '1':
            preprocess_with_pyannote(RAW_DATA_DIR, SEGMENTED_DATA_DIR, 5.0)
        else:
            preprocess_with_transcript(RAW_DATA_DIR, SEGMENTED_DATA_DIR, TRANSCRIPTION_DIR, 5.0)

    # 2. Extract Features
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at: {RAW_DATA_DIR}")
        return

    features_file_path = os.path.join(FEATURES_DIR, "features_dataset.xlsx")

    # Extract Text Features (Once)
    print("Extracting Text Features...")
    text_df = extract_text_features_from_dir(TRANSCRIPTION_DIR)

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
        # A. Audio Features
        base_features_file = extract_features(RAW_DATA_DIR, FEATURES_DIR)
        if not base_features_file:
            return

        if not text_df.empty:
            print("Merging Audio and Text Features")
            # Load Audio Features
            audio_df = pd.read_excel(base_features_file)
            
            audio_df['temp_id'] = audio_df['file_name'].apply(lambda x: os.path.splitext(x)[0])
            
            merged_df = pd.merge(audio_df, text_df, left_on='temp_id', right_on='file_name', how='left', suffixes=('', '_txt'))
            
            # Clean up keys
            if 'file_name_txt' in merged_df.columns:
                merged_df.drop(columns=['file_name_txt'], inplace=True)
            merged_df.drop(columns=['temp_id'], inplace=True)
            
            # Save back
            merged_df.to_excel(base_features_file, index=False)
            print(f"Merged features saved to {base_features_file}")
    else:
        base_features_file = features_file_path

    # 3. Initialize Experiments
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
        }#,
        # {
        #     "name": "Common Features (Intersection)",
        #     "use_filter": False,
        #     "use_selection": True,
        #     "suffix": "intersection"
        # },
        # {
        #     "name": "Combined (Filter + Selection)",
        #     "use_filter": True,
        #     "use_selection": True,
        #     "suffix": "combined"
        # }
    ]

    all_metrics = []

    # ==========================================
    # PIPELINE A: Segmented + GroupKFold
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE A: Segmented Audio + GroupKFold")
    print("="*50)
    
    feats_segmented = os.path.join(FEATURES_DIR, "features_segmented.xlsx")
    
    # A.1 Extract Segmented Features
    if not os.path.exists(feats_segmented):
         extract_features(SEGMENTED_DATA_DIR, FEATURES_DIR, "features_segmented.xlsx")
    else:
         print(f"Features already exist at {feats_segmented}")
         user_input = input("Do you want to overwrite and re-extract segmented features? (y/n): ").strip().lower()
         if user_input in ['y', 'yes']:
             print("Re-extracting features...")
             extract_features(SEGMENTED_DATA_DIR, FEATURES_DIR, "features_segmented.xlsx")
         else:
             print("Using existing segmented features.")

    # A.2 Merge Text Features into Segmented Features
    if os.path.exists(feats_segmented):
        print("Merging Text Features into Segmented Data...")
        seg_df = pd.read_excel(feats_segmented)
        
        if 'filler_ratio' not in seg_df.columns:
                seg_df['patient_id'] = seg_df['file_name'].astype(str).apply(lambda x: x.split('_')[0])
                
                merged_seg_df = pd.merge(seg_df, text_df, left_on='patient_id', right_on='file_name', how='left', suffixes=('', '_txt'))
                
                if 'file_name_txt' in merged_seg_df.columns:
                    merged_seg_df.drop(columns=['file_name_txt'], inplace=True)
                merged_seg_df.drop(columns=['patient_id'], inplace=True)
                
                merged_seg_df.to_excel(feats_segmented, index=False)
                print(f"Merged segmented features saved to {feats_segmented}")
        else:
            print("Text features already present in segmented data.")
    
    for exp in EXPERIMENTS:
        current_file = feats_segmented
        #A.3 Train and Evaluate
        metrics = train_and_evaluate(
            current_file, 
            RESULTS_DIR, 
            scenario_name=f"Prop_{exp['name']}", 
            validation_method="group_kfold",
            use_filter=exp["use_filter"],
            use_selection=exp["use_selection"]
        )
        all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE B: Baseline (Raw Audio + LOOCV)
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE B: Raw Audio + LOOCV")
    print("="*50)

    current_base = base_features_file

    for exp in EXPERIMENTS:
        current_file = current_base
        #B.1 Train and Evaluate
        metrics = train_and_evaluate(
            current_file, 
            RESULTS_DIR, 
            scenario_name=f"Base_{exp['name']}",
            validation_method="loocv",
            use_filter=exp["use_filter"],
            use_selection=exp["use_selection"]
        )
        all_metrics.extend(metrics)

    # 4. Consolidate Results
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