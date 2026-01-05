import numpy as np
import scipy.io.wavfile as wavfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def process_audio_file(input_wav, output_wav, n_speakers=2):
    try:
        # 1. Read Audio
        Fs, x = audioBasicIO.read_audio_file(input_wav)
        
        # Stereo to Mono
        if len(x.shape) > 1 and x.shape[1] > 1:
            x = np.mean(x, axis=1)

        # 2. Extract MID-TERM Features
        mt_win = 2.0
        mt_step = 0.2
        st_win = 0.05
        st_step = 0.05
        
        mt_features, st_features, _ = MidTermFeatures.mid_feature_extraction(
            x, Fs, 
            mt_win * Fs, 
            mt_step * Fs, 
            st_win * Fs, 
            st_step * Fs
        )
        
        X_features = mt_features.T
        X_features = np.nan_to_num(X_features)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        
        # 3. Cluster
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        flags = kmeans.fit_predict(X_scaled)
        
        # 4. Identify Patient (Speaker with most frames)
        unique, counts = np.unique(flags, return_counts=True)
        if len(unique) == 0:
             print(f"Warning: No clusters found for {input_wav}")
             return False
        
        patient_id = unique[np.argmax(counts)]
        
        # 5. Filter Signal
        step_samples = int(mt_step * Fs)
        mask = np.zeros(len(x), dtype=bool)
        
        for i, flag in enumerate(flags):
            start_sample = int(i * step_samples)
            end_sample = int(start_sample + step_samples)
            
            if start_sample >= len(x): break
            if end_sample > len(x): end_sample = len(x)
            
            if flag == patient_id:
                mask[start_sample:end_sample] = True
                
        patient_signal = x[mask]
        
        # 6. Save
        if len(patient_signal) == 0:
            print(f"Warning: Patient signal empty after processing {input_wav}")
            return False
            
        wavfile.write(output_wav, Fs, patient_signal.astype(np.int16))
        return True

    except Exception as e:
        print(f"Error processing {input_wav}: {e}")
        return False
