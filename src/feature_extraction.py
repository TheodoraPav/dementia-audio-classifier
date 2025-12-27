import os
import pandas as pd
from pyAudioAnalysis import MidTermFeatures

def extract_features(data_dir, output_dir):
    """
    Extracts mid-term features from audio files in the data directory.
    Assumes 'cc' (healthy) and 'cd' (dementia) subdirectories exist.
    """
    classes = {
        "healthy": {
            "path": os.path.join(data_dir, "cc"),
            "label": 0
        },
        "dementia": {
            "path": os.path.join(data_dir, "cd"),
            "label": 1
        }
    }

    mid_window = 2.0
    mid_step = 2.0
    short_window = 0.05
    short_step = 0.05
    compute_beat = False

    data_frames = []
    
    print("\n--- Starting Feature Extraction ---")

    for class_name, info in classes.items():
        dir_path = info["path"]
        label = info["label"]
        
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
            
        print(f"Processing {class_name}...")
        try:
            mid_term_features, wav_file_list, mid_feature_names = \
                MidTermFeatures.directory_feature_extraction(
                    dir_path, 
                    mid_window, 
                    mid_step, 
                    short_window, 
                    short_step, 
                    compute_beat=compute_beat
                )
            
            if len(mid_term_features) > 0:
                df = pd.DataFrame(mid_term_features, columns=mid_feature_names) 
                df['label'] = label
                df['file_name'] = [os.path.basename(f) for f in wav_file_list]
                data_frames.append(df)
                print(f"  -> Extracted {len(wav_file_list)} samples.")
        except Exception as e:
            print(f"  -> Error processing {class_name}: {e}")

    if len(data_frames) > 0:
        final_df = pd.concat(data_frames, ignore_index=True)
        
        # Reorder columns
        cols = ['file_name', 'label'] + [c for c in final_df.columns if c not in ['file_name', 'label']]
        final_df = final_df[cols]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, "features_dataset.xlsx")
        final_df.to_excel(output_file, index=False)
        print(f"Success! Features saved to {output_file}")
        return output_file
    else:
        print("No features extracted.")
        return None

if __name__ == "__main__":
    # Allow running as a standalone script for testing
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "Full_wave_enhanced_audio")
    OUTPUT_DIR = os.path.join(BASE_DIR, "features")
    extract_features(DATA_DIR, OUTPUT_DIR)
