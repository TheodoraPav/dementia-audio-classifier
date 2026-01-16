import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler

def parse_cha_file(file_path):

    stats = {
        'word_count': 0,
        'fillers_count': 0,
        'corrections_count': 0,
        'errors_count': 0,
        'total_duration': 0
    }
    
    current_speaker = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.strip()
                
                if line.startswith('*'):
                    if line.startswith('*PAR:'):
                        current_speaker = 'PAR'
                        content = line[5:].strip() # Remove '*PAR:'
                    else:
                        current_speaker = 'OTHER'
                        continue
                elif line.startswith('%') or line.startswith('@'):
                    current_speaker = None
                    continue
                elif current_speaker == 'PAR':
                    content = line_stripped
                else:
                    continue

                if current_speaker == 'PAR':
                    # Extract timestamps like 0_2360
                    timestamps = re.findall(r'\x15(\d+)_(\d+)\x15', content)
                    for start, end in timestamps:
                        stats['total_duration'] += (int(end) - int(start))
                    
                    # Remove timestamps like 0_2360
                    content = re.sub(r'\x15\d+_\d+\x15', '', content) 
                    content = re.sub(r'\d+_\d+', '', content)

                    # Count Fillers (&uh, &um, etc.)
                    # Match words starting with & followed by letters
                    fillers = re.findall(r'&[a-zA-Z]+', content)
                    stats['fillers_count'] += len(fillers)

                    # Count Corrections ([: replacement])
                    # Match [: text]
                    corrections = re.findall(r'\[:\s.*?\]', content)
                    stats['corrections_count'] += len(corrections)

                    # Count Errors ([* code])
                    # Match [* text]
                    errors = re.findall(r'\[\*\s.*?\]', content)
                    stats['errors_count'] += len(errors)

                    # Cleanup for Word Count
                    # Remove the markers we just counted
                    clean_text = re.sub(r'\[:\s.*?\]', '', content)
                    clean_text = re.sub(r'\[\*\s.*?\]', '', clean_text)
                    clean_text = re.sub(r'&[a-zA-Z]+', '', clean_text)
                    
                    # Remove other CHA markers like [//], [/], (.), etc.
                    # Remove brackets with content inside
                    clean_text = re.sub(r'\[.*?\]', '', clean_text)
                    # Remove special characters except spaces/alphanumerics
                    clean_text = re.sub(r'[^\w\s]', '', clean_text)
                    
                    words = [w for w in clean_text.split() if w.strip()]
                    stats['word_count'] += len(words)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    return stats

def extract_text_features_from_dir(transcription_dir):
    data = []
    
    subdirs = ['cc', 'cd']
    
    print(f"Scanning for CHA files in {transcription_dir}")

    for subdir in subdirs:
        dir_path = os.path.join(transcription_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith('.cha'):
                file_path = os.path.join(dir_path, filename)
                patient_id = os.path.splitext(filename)[0]
                
                stats = parse_cha_file(file_path)
                
                if stats:
                    total_words = stats['word_count'] if stats['word_count'] > 0 else 1
                    
                    row = {
                        'file_name': patient_id, # Linking key
                        'filler_ratio': stats['fillers_count'] / total_words,
                        'correction_ratio': stats['corrections_count'] / total_words,
                        'error_ratio': stats['errors_count'] / total_words
                    }

                    # Calculate words per minute
                    # total_duration is in milliseconds
                    duration_minutes = stats['total_duration'] / 60000.0
                    if duration_minutes > 0:
                        row['words_per_minute'] = stats['word_count'] / duration_minutes
                    else:
                        row['words_per_minute'] = 0
                    
                    data.append(row)

    if not data:
        print("No text features extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # Normalize the features
    scaler = StandardScaler()
    feature_cols = ['filler_ratio', 'correction_ratio', 'error_ratio', 'words_per_minute']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"Extracted and normalized text features for {len(df)} transcripts.")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transcription_path = os.path.join(current_dir, "..", "ADReSS-IS2020-train", "train", "transcription")
    
    df = extract_text_features_from_dir(transcription_path)
    print(df.head())
    print(df.describe())
