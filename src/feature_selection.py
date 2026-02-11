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
