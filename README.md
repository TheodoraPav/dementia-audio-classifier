# Medical Audio Classifier

A machine learning pipeline for detecting dementia from audio recordings.

## Project Structure
The project is organized into a modular pipeline:
*   `src/feature_extraction.py`: Wrapper for `pyAudioAnalysis` to extract acoustic features (MFCCs, etc.).
*   `src/preprocess.py`: Utilities for data cleaning, specifically removing highly correlated features.
*   `src/feature_selection.py`: Identifies features using the intersection of multiple models (RF, XGB, SVM).
*   `src/models.py`: Repository for model architectures and classifier configurations.
*   `src/train_model.py`: Module for model training, LOOCV execution, and metric reporting.
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

## Usage
Run the pipeline to extract features, select attributes, and classify:
```bash
python main.py
```

### Outputs
*   **Original Features**: `data/processed/features_dataset.xlsx`
*   **Filtered Features**: `data/processed/features_corr95_filtered.xlsx`
*   **Selected Features**: `data/processed/features_intersection_selected.xlsx`

## Roadmap
*   [ ] Feature Selection (Intersection of top features from RF/XGB/SVM)
*   [ ] Advanced Audio Processing (Speaker Diarization)
*   [ ] Model Training & Evaluation
