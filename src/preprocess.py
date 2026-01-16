import pandas as pd
import numpy as np
import os

def get_uncorrelated_features(df, threshold=0.95):
    features_df = df.select_dtypes(include=[np.number])
    
    cols_to_exclude = ['label', 'file_name']
    cols_to_analyze = [c for c in features_df.columns if c not in cols_to_exclude]
    
    features_df = features_df[cols_to_analyze]
    
    n_features_orig = features_df.shape[1]

    # Calculate Correlation
    corr_matrix = features_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Returns all columns from original df EXCEPT those dropped
    all_cols = df.columns.tolist()
    keep_cols = [c for c in all_cols if c not in to_drop]
    
    return keep_cols, to_drop

def remove_correlated_features(input_file, output_file, threshold=0.95):
    print(f"\nStarting Preprocessing (Correlation Threshold: {threshold})")
    
    # Load Data
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return None

    keep_cols, to_drop = get_uncorrelated_features(df, threshold)

    n_dropped = len(to_drop)
    print(f"Features to remove (> {threshold*100}% correlated): {n_dropped}")

    # Drop and Save
    df_cleaned = df[keep_cols]
    
    # Ensure output directory exists (though standard 'data/processed' usually does)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_cleaned.to_excel(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IN_FILE = os.path.join(BASE_DIR, "features", "features_dataset.xlsx")
    OUT_FILE = os.path.join(BASE_DIR, "features", "features_uncorrelated.xlsx")
    remove_correlated_features(IN_FILE, OUT_FILE)
