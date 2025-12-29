import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from src.models import get_models

def select_common_features(input_file, output_file, top_n=20):
    
    print(f"\n--- Starting Feature Selection (Intersection of Top {top_n}) ---")
    
    # 1. Load Data
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: file not found {input_file}")
        return None
        
    target_col = 'label'
    metadata_cols = ['file_name', target_col]
    
    X = df.drop(columns=metadata_cols, errors='ignore')
    y = df[target_col]
    feature_names = X.columns
    
    # Scale Data (Required for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train Models to get Importance
    print("Training models for feature importance...")
    
    models = get_models()
    
    # Random Forest
    rf = models["Random_Forest"]
    rf.fit(X, y)
    importances_rf = pd.Series(rf.feature_importances_, index=feature_names)
    top_rf = importances_rf.nlargest(top_n).index.tolist()
    
    # XGBoost
    xgb = models["XGBoost"]
    xgb.fit(X, y)
    importances_xgb = pd.Series(xgb.feature_importances_, index=feature_names)
    top_xgb = importances_xgb.nlargest(top_n).index.tolist()
    
    # SVM Linear (Coefficients)
    svm = models["SVM_Linear"]
    svm.fit(X_scaled, y)
    # Use absolute value of coefficients
    importances_svm = pd.Series(abs(svm.coef_[0]), index=feature_names)
    top_svm = importances_svm.nlargest(top_n).index.tolist()
    
    # 3. Calculate Intersection
    set_rf = set(top_rf)
    set_xgb = set(top_xgb)
    set_svm = set(top_svm)
    
    # Intersection of ALL 3
    common_features = list(set_rf.intersection(set_xgb).intersection(set_svm))
    
    print(f"Top {top_n} features selected by:")
    print(f"  - Random Forest")
    print(f"  - XGBoost")
    print(f"  - SVM Linear")
    
    print(f"Number of common features (Intersection): {len(common_features)}")
    
    # Fallback: If strict intersection is too small, try intersection of at least 2 models
    if len(common_features) < 3:
        print("Warning: Strict intersection yielded few features. Relaxing criteria (At least 2 models)...")
        # Items appearing in at least 2 sets
        all_features = top_rf + top_xgb + top_svm
        from collections import Counter
        counts = Counter(all_features)
        common_features = [f for f, count in counts.items() if count >= 2]
        print(f"Features in at least 2 models: {len(common_features)}")

    if len(common_features) == 0:
        print("Error: No common features found even with relaxed criteria. Returning None.")
        return None

    print(f"Selected Features: {common_features}")

    # 4. Save New Dataset
    # Keep metadata + selected features
    cols_to_keep = ['file_name', 'label'] + common_features
    df_selected = df[cols_to_keep]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_selected.to_excel(output_file, index=False)
    print(f"Saved feature subset to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IN_FILE = os.path.join(BASE_DIR, "features", "features_dataset.xlsx")
    OUT_FILE = os.path.join(BASE_DIR, "features", "features_intersection.xlsx")
    select_common_features(IN_FILE, OUT_FILE)
