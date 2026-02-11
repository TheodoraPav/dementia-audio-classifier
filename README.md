# Medical Audio Classifier

A comprehensive multimodal machine learning pipeline for early Alzheimer's dementia detection using audio recordings and linguistic transcripts from the ADReSS dataset. This system combines acoustic features (prosody, spectral characteristics) with linguistic markers (disfluencies, speech rate) to achieve robust, subject-independent dementia classification through multiple validation strategies and fusion approaches.

## Project Structure
The project is organized into a modular pipeline:
*   `src/preprocess_segments.py`: Handles audio segmentation using Pyannote diarization or transcript-based timestamps.
*   `src/feature_extraction.py`: Wrapper for `pyAudioAnalysis` to extract acoustic features (MFCCs, spectral features, etc.).
*   `src/merge_txt_feature.py`: Extracts linguistic features from `.cha` transcription files (disfluencies, speech rate).
*   `src/preprocess.py`: Utilities for data cleaning, specifically removing highly correlated features.
*   `src/feature_selection.py`: Provides dynamic feature selection using intersection of multiple models (RF, XGB, SVM) during training.
*   `src/models.py`: Repository for model architectures and classifier configurations.
*   `src/train_model.py`: Module for model training, cross-validation execution, and metric reporting.
*   `src/utils.py`: Helper functions for calculating performance metrics and feature importance visualization.
*   `src/late_fusion.py`: Implements late fusion strategy combining Audio-Only and Text-Only predictions.
*   `balance_segments_stratified.py`: Balances segmented dataset using stratified sampling to match median segment count.
*   `generate_report_charts.py`: Creates comparison charts for experimental results (14 charts total).
*   `generate_best_pipeline_summary.py`: Generates summary charts showing best performance per pipeline.
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

### 1. Run Main Pipeline
Execute the full training and evaluation pipeline:
```bash
python main.py
```
This will generate `outputs/final_consolidated_report.xlsx` containing all experimental results.

### 2. Generate Report Charts
Create comparison charts for thesis/publication:
```bash
python generate_report_charts.py
python generate_best_pipeline_summary.py
```
Charts are saved to `outputs/reported/` (150 DPI for fast compilation).

## Pipeline Structure

The system runs **8 pipelines** with **2 experiments** each (Baseline and Filtered Corr<0.95), evaluating **4 models** per pipeline:

### Pipelines
1. **Early Fusion (Segmented)** - Audio + Text features, Segmented audio, Leave-One-Group-Out validation
2. **Early Fusion (Raw)** - Audio + Text features, Raw audio, LOOCV validation
3. **TextOnly_Raw** - Text features only, Raw audio, LOOCV validation
4. **AudioOnly_Raw** - Audio features only, Raw audio, LOOCV validation
5. **AudioOnly_Segmented** - Audio features only, Segmented audio, Leave-One-Group-Out validation
6. **TextOnly_Segmented** - Text features only, Segmented audio, Leave-One-Group-Out validation
7. **Late Fusion (Raw)** - Weighted averaging of Audio-Only and Text-Only predictions (LOOCV)
8. **Continuous_Audio** - Audio + Text features from continuous patient-only audio, LOOCV validation

### Models
- SVM (Linear)
- SVM (RBF)
- Random Forest
- XGBoost

### Outputs
*   **Processed Features**: `data/processed/features_*.xlsx` (tracked in Git)
*   **Consolidated Report**: `outputs/final_consolidated_report.xlsx` (used by report generation scripts)
*   **SVM Feature Importance**: `outputs/*_final_svm_features.png` and `*_final_svm_weights.csv`
*   **Comparison Charts**: `outputs/reported/*.png` (14 charts for experimental analysis)
*   **Predictions**: `outputs/predictions/preds_*.csv` (intermediate predictions for Late Fusion)
*   **Late Fusion Results**: `outputs/late_fusion/fused_predictions.csv`

## Key Features

### 1. Advanced Audio Preprocessing
The pipeline supports three methods for processing patient audio:
*   **Pyannote Diarization**: Uses speaker diarization to separate patient speech into 5-second segments.
*   **Transcript-Based Segmentation**: Uses `.cha` transcription files for exact timestamp extraction into 5-second segments.
*   **Continuous Patient Audio**: Merges all patient speech segments into one continuous audio file per patient.

### 2. Feature Extraction
*   **Audio Features**: Acoustic features using `pyAudioAnalysis` (MFCCs, spectral features, chroma excluded).
*   **Text Features**: Linguistic markers from transcriptions including:
    - Filler ratio
    - Pause ratio
    - Repetition ratio
    - Error ratio
    - Correction ratio
    - Self-correction ratio
    - Words per minute (speech rate based on `.cha` timestamps)

### 3. Multimodal Fusion Strategies
*   **Early Fusion**: Concatenates audio and text features before classification.
*   **Late Fusion**: Combines predictions from independent audio and text models using weighted averaging.

### 4. Validation Methodologies
*   **LOOCV (Leave-One-Out)**: For raw, full-length audio to ensure subject-independent evaluation.
*   **LOGO (Leave-One-Group-Out)**: For segmented audio to prevent data leakage across segments from the same subject.

### 5. Feature Selection
*   **Correlation Filtering**: Removes redundant features with high correlation (>95%).
*   **Baseline Comparison**: Evaluates performance with and without filtering.

### 6. Works Without Raw Data
If raw data is missing, the pipeline uses pre-computed features from `data/processed/`.

## Performance Summary
According to experimental results:
- **Best F1-Score**: Text-only features (0.747)
- **Best AUC-ROC**: Late Fusion with SVM Linear (0.794)
- **Optimal Configuration**: Late Fusion on raw audio with subject-independent validation
