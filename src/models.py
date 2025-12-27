from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        "SVM_Linear": SVC(
            kernel='linear', 
            C=1.0, 
            probability=True, 
            random_state=42
        ),
        "SVM_RBF": SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            probability=True, 
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42,
            eval_metric='logloss'
        )
    }
    return models
