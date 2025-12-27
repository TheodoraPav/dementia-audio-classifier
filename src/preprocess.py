import pandas as pd
import numpy as np
import os

def remove_correlated_features(input_file, output_file, threshold=0.95):
    """
    Reads a feature dataset, removes features with correlation > threshold,
    and saves the cleaned dataset.
    """
    print(f"\n--- Starting Preprocessing (Correlation Threshold: {threshold}) ---")
    
    # 1. Load Data
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return None

    # Identify metadata columns to preserve
    metadata_cols = ['file_name', 'label']
    features_df = df.drop(columns=metadata_cols, errors='ignore')
    
    # Get numeric features only (just in case)
    features_df = features_df.select_dtypes(include=[np.number])
    
    n_features_orig = features_df.shape[1]
    print(f"Original feature count: {n_features_orig}")

    # 2. Calculate Correlation
    corr_matrix = features_df.corr().abs()

    # 3. Identify features to drop
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    n_dropped = len(to_drop)
    print(f"Features to remove (> {threshold*100}% correlated): {n_dropped}")

    # 4. Drop and Save
    df_cleaned = df.drop(columns=to_drop)
    
    # Ensure output directory exists (though standard 'data/processed' usually does)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_cleaned.to_excel(output_file, index=False)
    print(f"Remaining features: {df_cleaned.shape[1] - len(metadata_cols)}")
    print(f"Cleaned dataset saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IN_FILE = os.path.join(BASE_DIR, "features", "features_dataset.xlsx")
    OUT_FILE = os.path.join(BASE_DIR, "features", "features_uncorrelated.xlsx")
    remove_correlated_features(IN_FILE, OUT_FILE)
