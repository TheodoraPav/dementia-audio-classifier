# Medical Audio Classifier

A machine learning pipeline for detecting dementia from audio recordings.

## Project Structure
The project is organized into a modular pipeline:
*   `src/audio_processing.py`: Implements K-Means clustering for speaker diarization.
*   `src/preprocess_segments.py`: Handles audio segmentation based on diarization results.
*   `src/feature_extraction.py`: Wrapper for `pyAudioAnalysis` to extract acoustic features (MFCCs, etc.).
*   `src/merge_txt_feature.py`: Extracts linguistic features from `.cha` transcription files.
*   `src/preprocess.py`: Utilities for data cleaning, specifically removing highly correlated features.
*   `src/feature_selection.py`: Identifies features using the intersection of multiple models (RF, XGB, SVM).
*   `src/models.py`: Repository for model architectures and classifier configurations.
*   `src/train_model.py`: Module for model training, cross-validation execution, and metric reporting.
*   `src/utils.py`: Helper functions for calculating performance metrics and generating plots.
*   `src/late_fusion.py`: Implements late fusion strategy combining Audio-Only and Text-Only predictions.
*   `balance_segments_stratified.py`: Balances segmented dataset using stratified sampling.
*   `main.py`: The entry point script that orchestrates the full pipeline.

## Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Install `pyAudioAnalysis`:
    ```bash
    git clone https://github.com/tyiannak/pyAudioAnalysis.git
    pip install -e pyAudioAnalysis
    ```

### Compatibility Warning 
* Python Version: Python 3.10 or 3.11

*Technical Requirement*: This project depends on `pyAudioAnalysis`, which requires the aifc module. 
As of Python 3.13, aifc was removed from the standard library. To avoid installation errors, please ensure you are using Python 3.10 or 3.11.

## Usage
Run the pipeline to extract features, select attributes, and classify:
```bash
python main.py
```

## Pipeline Structure

The system runs **8 pipelines** with **2 experiments** each (Baseline and Filtered), evaluating **4 models** per pipeline:

### Pipelines
1. **Combined_Segmented** - Audio + Text features, Segmented audio, Leave-One-Group-Out validation
2. **Combined_Raw** - Audio + Text features, Raw audio, LOOCV validation
3. **TextOnly_Raw** - Text features only, Raw audio, LOOCV validation
4. **AudioOnly_Raw** - Audio features only, Raw audio, LOOCV validation
5. **AudioOnly_Segmented** - Audio features only, Segmented audio, Leave-One-Group-Out validation
6. **TextOnly_Segmented** - Text features only, Segmented audio, Leave-One-Group-Out validation
7. **Late_Fusion** - Combined predictions from Audio-Only and Text-Only models (Raw audio, LOOCV)
8. **Continuous_Audio** - Audio + Text features from continuous patient audio, LOOCV validation

### Models
- SVM (Linear)
- SVM (RBF)
- Random Forest
- XGBoost

### Outputs
*   **Processed Features**: `data/processed/features_*.xlsx` (tracked in Git)
*   **Consolidated Report**: `outputs/final_consolidated_report.xlsx`
*   **ROC Curves**: `outputs/roc_*.png` (per-scenario)
*   **Results Tables**: `outputs/results_*.xlsx` (per-scenario)
*   **SVM Feature Importance**: `outputs/*_final_svm_features.png` and `*_final_svm_weights.csv`
*   **Global Performance Charts**: `outputs/global_metrics_bar_charts.png` and `outputs/auc_comparison.png`
*   **Predictions**: `outputs/predictions/preds_*.csv` (for Late Fusion)
*   **Late Fusion Results**: `outputs/late_fusion/fused_predictions.csv`

## Key Features

### 1. Advanced Audio Preprocessing
The pipeline supports three methods for processing patient audio:
*   **Pyannote Diarization**: Uses speaker diarization to separate patient speech into 5-second segments.
*   **Transcript-Based Segmentation**: Uses `.cha` transcription files for exact timestamp extraction into 5-second segments.
*   **Continuous Patient Audio**: Merges all patient speech segments into one continuous audio file per patient.

### 2. Feature Extraction
*   **Audio Features**: Acoustic features using `pyAudioAnalysis` (MFCCs, spectral features, etc.).
*   **Text Features**: Linguistic markers from transcriptions including:
    - Filler ratio (`um`, `uh`)
    - Pause ratio (`&=laugh`, `&=cough`, etc.)
    - Repetition ratio (`[/]`)
    - Error ratio (`[*]`)
    - Correction ratio (`[//]`)
    - Self-correction ratio (utterances with self-corrections)
    - Words per minute (speech rate)

### 3. Feature Selection
*   **Remove Correlated Features**: Eliminates redundant features with high correlation (>95%).
*   **Select Top Features**: Keeps only the most relevant features identified by multiple models.

### 4. Works Without Raw Data
If raw data is missing, the pipeline uses pre-computed features from `data/processed/` (tracked in Git).

## Roadmap
*   [x] Feature Selection (Intersection of top features from RF/XGB/SVM)
*   [x] Advanced Audio Processing (Speaker Diarization)
*   [x] Model Training & Evaluation
*   [x] Multi-Pipeline Comparison (Feature combinations and validation methods)
