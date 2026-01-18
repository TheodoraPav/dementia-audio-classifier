import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.models import get_models

def get_selected_features(X, y, top_n=30):
    print(f"Running feature selection (Top {top_n})")
    
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = get_models()
    
    rf = models["Random_Forest"]
    rf.fit(X, y)
    importances_rf = pd.Series(rf.feature_importances_, index=feature_names)
    top_rf = importances_rf.nlargest(top_n).index.tolist()
    
    xgb = models["XGBoost"]
    xgb.fit(X, y)
    importances_xgb = pd.Series(xgb.feature_importances_, index=feature_names)
    top_xgb = importances_xgb.nlargest(top_n).index.tolist()
    
    svm = models["SVM_Linear"]
    svm.fit(X_scaled, y)
    importances_svm = pd.Series(abs(svm.coef_[0]), index=feature_names)
    top_svm = importances_svm.nlargest(top_n).index.tolist()
    
    set_rf = set(top_rf)
    set_xgb = set(top_xgb)
    set_svm = set(top_svm)
    
    # Intersection of ALL 3
    common_features =list(set_rf.intersection(set_xgb).intersection(set_svm))
    
    # Validation Threshold
    MIN_FEATURES = 5

    # 1. Relaxed Criteria (At least 2 models)
    if len(common_features) < MIN_FEATURES:
        print(f"Warning: Strict intersection yielded {len(common_features)} features. Relaxing criteria (At least 2 models)...")
        all_features = top_rf + top_xgb + top_svm
        from collections import Counter
        counts = Counter(all_features)
        common_features = [f for f, count in counts.items() if count >= 2]
    
    # 2. Fallback Criteria (Random Forest Top N)
    if len(common_features) < MIN_FEATURES:
        print(f"Warning: Relaxed intersection yielded {len(common_features)} features. Fallback to Random Forest features.")
        common_features = top_rf
    
    return common_features

def select_common_features(input_file, output_file, top_n=30):
    
    print(f"\nStarting Feature Selection (Intersection of Top {top_n})")
    
    #1. Load Data
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: file not found {input_file}")
        return None
        
    target_col = 'label'
    metadata_cols = ['file_name', target_col]
    
    X = df.drop(columns=metadata_cols, errors='ignore')
    y = df[target_col]
    
    #2. Train Models to get Importance
    common_features = get_selected_features(X, y, top_n=top_n)

    if len(common_features) == 0:
        print("Error: No common features found even with relaxed criteria. Returning None.")
        return None

    print(f"Selected Features: {common_features}")

    #3. Save New Dataset
    cols_to_keep = ['file_name', 'label'] + common_features
    df_selected = df[cols_to_keep]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_selected.to_excel(output_file, index=False)
    print(f"Saved feature subset to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IN_FILE = os.path.join(BASE_DIR, "data", "processed", "features_dataset.xlsx")
    OUT_FILE = os.path.join(BASE_DIR, "data", "processed", "features_intersection.xlsx")
    select_common_features(IN_FILE, OUT_FILE)
