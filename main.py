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
        if not os.path.exists(RAW_DATA_DIR):
            print(f"Warning: Raw data directory not found at: {RAW_DATA_DIR}")
            print("Skipping preprocessing. Will use existing segmented data if available.")
            run_preprocessing = False
    
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
    features_file_path = os.path.join(FEATURES_DIR, "features_dataset.xlsx")

    # Check if raw data exists
    raw_data_exists = os.path.exists(RAW_DATA_DIR)
    if not raw_data_exists:
        print(f"\nWarning: Raw data directory not found at: {RAW_DATA_DIR}")
        print("Will attempt to use existing processed features if available.")

    # Extract Text Features (Only if transcription directory exists)
    text_df = pd.DataFrame()
    if os.path.exists(TRANSCRIPTION_DIR):
        print("Extracting Text Features...")
        text_df = extract_text_features_from_dir(TRANSCRIPTION_DIR)
    else:
        print(f"Warning: Transcription directory not found at: {TRANSCRIPTION_DIR}")
        print("Skipping text feature extraction.")

    should_extract = True
    if os.path.exists(features_file_path):
        print(f"Features file found at: {features_file_path}")
        if raw_data_exists:
            user_response = input("Do you want to overwrite it and re-extract features? (y/n): ").strip().lower()
            if user_response != 'y':
                should_extract = False
                print("Using existing features.")
            else:
                print("Re-extracting features...")
        else:
            should_extract = False
            print("Raw data not available. Using existing features.")
    elif not raw_data_exists:
        print("Error: No processed features found and raw data is missing.")
        print("Please either:")
        print("  1. Add the raw data to ADReSS-IS2020-train/, OR")
        print("  2. Pull the processed features from Git (git pull)")
        return
    

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
        }
    ]

    all_metrics = []

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
    
    # A.3 Balance Segmented Features (stratified sampling based on median)
    feats_segmented_balanced = os.path.join(FEATURES_DIR, "features_segmented_balanced.xlsx")
    
    if os.path.exists(feats_segmented) and not os.path.exists(feats_segmented_balanced):
        print("\nBalancing segmented features (stratified sampling, target=median)...")
        from balance_segments_stratified import balance_segments_stratified
        balance_segments_stratified(feats_segmented, feats_segmented_balanced, 'median')
    elif os.path.exists(feats_segmented_balanced):
        print(f"Balanced segmented features already exist at {feats_segmented_balanced}")
            
    # ==========================================
    # PIPELINE A: Segmented + Leave-One-Group-Out
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE A: Segmented Audio + Leave-One-Group-Out")
    print("="*50)
    for exp in EXPERIMENTS:
        current_file = feats_segmented_balanced  # Use balanced version
        #A.4 Train and Evaluate
        metrics = train_and_evaluate(
            current_file, 
            RESULTS_DIR, 
            scenario_name=f"Combined_Segmented_{exp['name']}", 
            validation_method="leave_one_group_out",
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
            scenario_name=f"Combined_Raw_{exp['name']}",
            validation_method="loocv",
            use_filter=exp["use_filter"],
            use_selection=exp["use_selection"]
        )
        all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE C: Text-Only Features + LOOCV
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE C: Text-Only Features + LOOCV")
    print("="*50)
    
    feats_text_only = os.path.join(FEATURES_DIR, "features_text_only.xlsx")
    
    if os.path.exists(base_features_file):
        df_full = pd.read_excel(base_features_file)
        text_feature_cols = ['filler_ratio', 'pause_ratio', 'rep_ratio', 'error_ratio', 
                            'correction_ratio', 'self_correction_ratio', 'words_per_minute']
        
        required_cols = ['file_name', 'label'] + text_feature_cols
        df_text = df_full[required_cols]
        df_text.to_excel(feats_text_only, index=False)
        print(f"Text-only features saved to {feats_text_only}")
    else:
        print(f"Error: Base features file not found at {base_features_file}")
    
    if os.path.exists(feats_text_only):
        for exp in EXPERIMENTS:
            current_file = feats_text_only
            metrics = train_and_evaluate(
                current_file, 
                RESULTS_DIR, 
                scenario_name=f"TextOnly_Raw_{exp['name']}", 
                validation_method="loocv",
                use_filter=exp["use_filter"],
                use_selection=exp["use_selection"]
            )
            all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE D: Audio-Only Features + LOOCV
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE D: Audio-Only Features + LOOCV")
    print("="*50)
    
    feats_audio_only = os.path.join(FEATURES_DIR, "features_audio_only.xlsx")
    
    if os.path.exists(base_features_file):
        df_full = pd.read_excel(base_features_file)
        text_feature_cols = ['filler_ratio', 'pause_ratio', 'rep_ratio', 'error_ratio', 
                            'correction_ratio', 'self_correction_ratio', 'words_per_minute']
        
        audio_cols = [col for col in df_full.columns if col not in text_feature_cols]
        df_audio = df_full[audio_cols]
        df_audio.to_excel(feats_audio_only, index=False)
        print(f"Audio-only features saved to {feats_audio_only}")
    else:
        print(f"Error: Base features file not found at {base_features_file}")
    
    if os.path.exists(feats_audio_only):
        for exp in EXPERIMENTS:
            current_file = feats_audio_only
            metrics = train_and_evaluate(
                current_file, 
                RESULTS_DIR, 
                scenario_name=f"AudioOnly_Raw_{exp['name']}", 
                validation_method="loocv",
                use_filter=exp["use_filter"],
                use_selection=exp["use_selection"]
            )
            all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE E: Audio-Only + Segmented + Leave-One-Group-Out
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE E: Audio-Only + Segmented + Leave-One-Group-Out")
    print("="*50)
    
    feats_audio_seg = os.path.join(FEATURES_DIR, "features_audio_segmented_balanced.xlsx")
    
    if os.path.exists(feats_segmented_balanced):
        df_seg = pd.read_excel(feats_segmented_balanced)
        text_feature_cols = ['filler_ratio', 'pause_ratio', 'rep_ratio', 'error_ratio', 
                            'correction_ratio', 'self_correction_ratio', 'words_per_minute']
        
        audio_cols = [col for col in df_seg.columns if col not in text_feature_cols]
        df_audio_seg = df_seg[audio_cols]
        df_audio_seg.to_excel(feats_audio_seg, index=False)
        print(f"Audio-only segmented features saved to {feats_audio_seg}")
    else:
        print(f"Error: Balanced segmented features file not found at {feats_segmented_balanced}")
    
    if os.path.exists(feats_audio_seg):
        for exp in EXPERIMENTS:
            current_file = feats_audio_seg
            metrics = train_and_evaluate(
                current_file, 
                RESULTS_DIR, 
                scenario_name=f"AudioOnly_Segmented_{exp['name']}", 
                validation_method="leave_one_group_out",
                use_filter=exp["use_filter"],
                use_selection=exp["use_selection"]
            )
            all_metrics.extend(metrics)

    # ==========================================
    # PIPELINE F: Text-Only + Segmented + Leave-One-Group-Out
    # ==========================================
    print("\n" + "="*50)
    print("PIPELINE F: Text-Only + Segmented + Leave-One-Group-Out")
    print("="*50)
    
    feats_text_seg = os.path.join(FEATURES_DIR, "features_text_segmented_balanced.xlsx")
    
    if os.path.exists(feats_segmented_balanced):
        df_seg = pd.read_excel(feats_segmented_balanced)
        text_feature_cols = ['filler_ratio', 'pause_ratio', 'rep_ratio', 'error_ratio', 
                            'correction_ratio', 'self_correction_ratio', 'words_per_minute']
        
        required_cols = ['file_name', 'label'] + text_feature_cols
        df_text_seg = df_seg[required_cols]
        df_text_seg.to_excel(feats_text_seg, index=False)
        print(f"Text-only segmented features saved to {feats_text_seg}")
    else:
        print(f"Error: Balanced segmented features file not found at {feats_segmented_balanced}")
    
    if os.path.exists(feats_text_seg):
        for exp in EXPERIMENTS:
            current_file = feats_text_seg
            metrics = train_and_evaluate(
                current_file, 
                RESULTS_DIR, 
                scenario_name=f"TextOnly_Segmented_{exp['name']}", 
                validation_method="leave_one_group_out",
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