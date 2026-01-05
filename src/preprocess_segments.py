import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from src.audio_processing import process_audio_file

def segment_audio(input_wav, output_dir, file_basename, segment_duration=10.0):
    try:
        sample_rate, data = wavfile.read(input_wav)
        
        # Calculate number of samples per segment
        samples_per_segment = int(segment_duration * sample_rate)
        total_samples = len(data)
        
        if total_samples == 0:
            print(f"Warning: Empty audio file {input_wav}")
            return
            
        num_segments = total_samples // samples_per_segment
        
        count = 0
        for i in range(num_segments + 1):
            start = i * samples_per_segment
            end = start + samples_per_segment
            
            if start >= total_samples:
                break
                
            # Handle last chunk
            if end > total_samples:
                end = total_samples
                
            # Skip if chunk is too short (< 1s)
            if (end - start) < (1.0 * sample_rate):
                continue
                
            chunk = data[start:end]
            
            # Naming convention: id01_1.wav, id01_2.wav
            # Indices are 1-based as per example
            out_name = f"{file_basename}_{count + 1}.wav"
            out_path = os.path.join(output_dir, out_name)
            
            wavfile.write(out_path, sample_rate, chunk)
            count += 1
            
        return count
        
    except Exception as e:
        print(f"Error segmenting {input_wav}: {e}")
        return 0

def preprocess_dataset(raw_data_dir, output_dir, segment_duration=5.0):

    print("==================================================")
    print("   Audio Preprocessing: Diarization & Segmentation")
    print("==================================================")
    print(f"Chunk Duration: {segment_duration}s")
    
    CLASSES = ["cc", "cd"] # Healthy, Dementia
    
    temp_dir = os.path.join(os.path.dirname(output_dir), "temp_diarized")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    for cls in CLASSES:
        in_path = os.path.join(raw_data_dir, cls)
        out_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(in_path):
            print(f"Warning: Class folder not found: {in_path}")
            continue
            
        os.makedirs(out_path, exist_ok=True)
        
        files = [f for f in os.listdir(in_path) if f.endswith(".wav")]
        print(f"\nProcessing '{cls}' ({len(files)} files)...")
        
        total_chunks = 0
        
        for f in files:
            in_file = os.path.join(in_path, f)
            basename = os.path.splitext(f)[0]
            
            # Temporary path for diarized file
            temp_diarized_file = os.path.join(temp_dir, f"{basename}_diarized.wav")
            
            # 1. Diarization (Remove Interviewer)
            success = process_audio_file(in_file, temp_diarized_file)
            
            if success:
                # 2. Segmentation
                n_chunks = segment_audio(temp_diarized_file, out_path, basename, segment_duration=segment_duration)
                total_chunks += n_chunks
                
                # Cleanup temp file
                if os.path.exists(temp_diarized_file):
                    os.remove(temp_diarized_file)
            else:
                print(f"Skipped {f} (Diarization failed)")
                
        print(f"Generated {total_chunks} chunks for '{cls}'.")
        
    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except:
        pass
        
    print("\n========================================")
    print("Preprocessing Complete!")
    print(f"Segmented files saved in: {output_dir}")
    print("========================================")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input: Raw Audio
    RAW_DATA_DIR = os.path.join(BASE_DIR, "ADReSS-IS2020-train", "train", "Full_wave_enhanced_audio")
    
    # Output: Segmented Audio
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "segmented_audio")
    
    preprocess_dataset(RAW_DATA_DIR, OUTPUT_DIR)
