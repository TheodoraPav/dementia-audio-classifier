import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
try:
    from pyannote.audio import Pipeline
except ImportError:
    print("Warning: pyannote.audio not installed. 'preprocess_with_pyannote' will fail if called.")
    Pipeline = None


def preprocess_with_pyannote(raw_data_dir, output_dir, segment_duration=5.0):
    print("=== Audio Preprocessing V2: Pyannote Diarization ===")
    
    try:
        # Load pipeline (ASSUMES AUTH TOKEN IS SET via huggingface-cli login)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
    except Exception as e:
        print(f"Error loading Pyannote pipeline: {e}")
        return

    CLASSES = ["cc", "cd"]
    temp_dir = os.path.join(os.path.dirname(output_dir), "temp_diarized_v2")
    os.makedirs(temp_dir, exist_ok=True)
        
    for cls in CLASSES:
        in_path = os.path.join(raw_data_dir, cls)
        out_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(in_path): continue
        os.makedirs(out_path, exist_ok=True)
        
        files = [f for f in os.listdir(in_path) if f.endswith(".wav")]
        print(f"\nProcessing '{cls}' ({len(files)} files)...")
        total_chunks = 0
        
        for f in files:
            in_file = os.path.join(in_path, f)
            basename = os.path.splitext(f)[0]
            
            try:
                # 1. Run Diarization
                diarization = pipeline(in_file)
                
                # 2. Read Audio (using scipy, already imported)
                sample_rate, data = wavfile.read(in_file)
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # 3. Collect Speaker Stats
                speakers = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speakers:
                        speakers[speaker] = {'duration': 0, 'energy_sum': 0, 'segments': []}
                    
                    start = int(turn.start * sample_rate)
                    end = int(turn.end * sample_rate)
                    chunk = data[start:end]
                    
                    if len(chunk) == 0: continue

                    # RMS Calculation (numpy)
                    rms = np.sqrt(np.mean(chunk**2))
                    dur = turn.end - turn.start
                    
                    speakers[speaker]['duration'] += dur
                    speakers[speaker]['energy_sum'] += rms * dur
                    speakers[speaker]['segments'].append(chunk)

                if not speakers: continue

                # 4. Identify Patient (Max Duration & Max Loudness)
                for s in speakers:
                    tot_dur = speakers[s]['duration']
                    speakers[s]['avg_rms'] = speakers[s]['energy_sum'] / tot_dur if tot_dur > 0 else 0

                best_speaker = max(speakers, key=lambda s: speakers[s]['duration'])
                
                # 5. Reconstruct & Save
                patient_audio = np.concatenate(speakers[best_speaker]['segments'])
                
                # Save as single file (No segmentation requested)
                final_out_file = os.path.join(out_path, f"{basename}.wav")
                wavfile.write(final_out_file, sample_rate, patient_audio.astype(np.int16))
                total_chunks += 1
                

            except Exception as e:
                print(f"Error processing {f}: {e}")
                
        print(f"Generated {total_chunks} chunks for '{cls}'.")
        
    try: os.rmdir(temp_dir)
    except: pass
    print("Preprocessing V2 Complete!")

# ==========================================================
# Transcript-Based Preprocessing
# ==========================================================
import re

def parse_timestamps_v3(line):
    # Try \x15 pattern first
    match = re.search(r'\x15(\d+)_(\d+)\x15', line)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try simple numeric at end
    parts = line.strip().split()
    if parts:
        last = parts[-1]
        match = re.match(r'(\d+)_(\d+)', last)
        if match:
            return int(match.group(1)), int(match.group(2))
            
    return None, None

def preprocess_with_transcript(raw_data_dir, output_dir, transcription_dir, segment_duration=5.0):
    print("==================================================")
    print("   Audio Preprocessing: Transcript-Based     ")
    print("==================================================")
    
    CLASSES = ["cc", "cd"]
    
    for cls in CLASSES:
        in_path = os.path.join(raw_data_dir, cls)
        trans_path = os.path.join(transcription_dir, cls)
        out_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(in_path): continue
        os.makedirs(out_path, exist_ok=True)
        
        files = [f for f in os.listdir(in_path) if f.endswith(".wav")]
        print(f"\nProcessing '{cls}' ({len(files)} files)...")
        total_chunks = 0
        
        for f in files:
            basename = os.path.splitext(f)[0]
            wav_file = os.path.join(in_path, f)
            cha_file = os.path.join(trans_path, f"{basename}.cha")
            
            if not os.path.exists(cha_file):
                print(f"Skipping {f}: Transcript not found at {cha_file}")
                continue
                
            try:
                # Read Audio
                sample_rate, audio_data = wavfile.read(wav_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1) # Mono
                
                with open(cha_file, 'r', encoding='utf-8') as cf:
                    lines = cf.readlines()
                    
                # 1. Identify Runs of consecutive PAR segments
                merged_runs = [] # List of {'start': ms, 'end': ms}
                current_run = None
                
                # Debug: specific file
                is_debug = (basename == 'S001')
                if is_debug: print(f"DEBUG: Parsing {cha_file}")

                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    if line.startswith("*PAR:"):
                        s, e = parse_timestamps_v3(line)
                        if s is None:
                             if i + 1 < len(lines) and lines[i+1].strip().startswith("%snd"):
                                 s, e = parse_timestamps_v3(lines[i+1])
                        
                        if s is not None:
                            if current_run:
                                current_run['end'] = e
                            else:
                                current_run = {'start': s, 'end': e}
                                merged_runs.append(current_run)
                                
                    elif line.startswith("*INV:"):
                        # Investigator speaks. 
                        # Check if we have a timestamp for this INV line to close the PAR run precisely.
                        s, e = parse_timestamps_v3(line)
                        if s is None:
                             if i + 1 < len(lines) and lines[i+1].strip().startswith("%snd"):
                                 s, e = parse_timestamps_v3(lines[i+1])
                        
                        if current_run:
                            if s is not None and s > current_run['start']:
                                current_run['end'] = s
                            
                            # End the run
                            current_run = None
                        
                # 2. Extract and Split
                file_chunk_count = 0
                
                # 2. Extract and Split
                file_chunk_count = 0
                
                for run in merged_runs:
                    start_sample = int(run['start'] * sample_rate / 1000)
                    end_sample = int(run['end'] * sample_rate / 1000)
                    
                    if start_sample >= len(audio_data): continue
                    if end_sample > len(audio_data): end_sample = len(audio_data)
                    
                    if end_sample <= start_sample: continue

                    run_audio = audio_data[start_sample:end_sample]
                    
                    # Split into chunks
                    chunk_samples = int(segment_duration * sample_rate)
                    
                    if len(run_audio) < chunk_samples:
                        # Keep it even if small (User req: "exactly from... to...")
                        out_name = f"{basename}_{file_chunk_count}.wav"
                        out_full = os.path.join(out_path, out_name)
                        wavfile.write(out_full, sample_rate, run_audio.astype(np.int16))
                        file_chunk_count += 1
                    else:
                        num_splits = len(run_audio) // chunk_samples
                        for k in range(num_splits + 1):
                            c_start = k * chunk_samples
                            c_end = c_start + chunk_samples
                            
                            if c_start >= len(run_audio): break
                            if c_end > len(run_audio): c_end = len(run_audio)
                            
                            if (c_end - c_start) <= 0: continue
                            
                            chunk_data = run_audio[c_start:c_end]
                            out_name = f"{basename}_{file_chunk_count}.wav"
                            out_full = os.path.join(out_path, out_name)
                            wavfile.write(out_full, sample_rate, chunk_data.astype(np.int16))
                            file_chunk_count += 1
                            
                total_chunks += file_chunk_count
                
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
        print(f"Generated {total_chunks} chunks for '{cls}'.")


# ==========================================================
# Continuous Patient Audio (No Chunking)
# ==========================================================

def preprocess_continuous_patient_audio(raw_data_dir, output_dir, transcription_dir):
    print("==================================================")
    print("   Audio Preprocessing: Continuous Patient Audio")
    print("==================================================")
    
    CLASSES = ["cc", "cd"]
    
    for cls in CLASSES:
        in_path = os.path.join(raw_data_dir, cls)
        trans_path = os.path.join(transcription_dir, cls)
        out_path = os.path.join(output_dir, cls)
        
        if not os.path.exists(in_path): 
            continue
        os.makedirs(out_path, exist_ok=True)
        
        files = [f for f in os.listdir(in_path) if f.endswith(".wav")]
        print(f"\nProcessing '{cls}' ({len(files)} files)...")
        total_files = 0
        
        for f in files:
            basename = os.path.splitext(f)[0]
            wav_file = os.path.join(in_path, f)
            cha_file = os.path.join(trans_path, f"{basename}.cha")
            
            if not os.path.exists(cha_file):
                print(f"Skipping {f}: Transcript not found at {cha_file}")
                continue
                
            try:
                # Read Audio
                sample_rate, audio_data = wavfile.read(wav_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                
                with open(cha_file, 'r', encoding='utf-8') as cf:
                    lines = cf.readlines()
                    
                # Collect all patient segments
                patient_segments = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    if line.startswith("*PAR:"):
                        s, e = parse_timestamps_v3(line)
                        if s is None:
                            # Check next line for timestamp
                            if i + 1 < len(lines) and lines[i+1].strip().startswith("%snd"):
                                s, e = parse_timestamps_v3(lines[i+1])
                        
                        if s is not None:
                            # Convert ms to samples
                            start_sample = int(s * sample_rate / 1000)
                            end_sample = int(e * sample_rate / 1000)
                            
                            # Validate bounds
                            if start_sample >= len(audio_data):
                                continue
                            if end_sample > len(audio_data):
                                end_sample = len(audio_data)
                            if end_sample <= start_sample:
                                continue
                            
                            # Extract segment
                            segment = audio_data[start_sample:end_sample]
                            patient_segments.append(segment)
                
                # Concatenate all patient segments
                if patient_segments:
                    continuous_audio = np.concatenate(patient_segments)
                    
                    # Save as single continuous file
                    out_name = f"{basename}.wav"
                    out_full = os.path.join(out_path, out_name)
                    wavfile.write(out_full, sample_rate, continuous_audio.astype(np.int16))
                    total_files += 1
                    print(f"  Created continuous audio for {basename}: {len(continuous_audio)/sample_rate:.2f}s")
                else:
                    print(f"  Warning: No patient segments found for {basename}")
                    
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
        print(f"Generated {total_files} continuous audio files for '{cls}'.")
    
    print("\nContinuous Patient Audio Preprocessing Complete!")

