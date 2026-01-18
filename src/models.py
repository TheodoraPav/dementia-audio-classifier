from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Prioritize Class 1 (Dementia) by applying a higher penalty to False Negatives.
# This forces the models to be more "sensitive" to patients, boosting Recall.
custom_weights = {0: 1, 1: 2.5}

def get_models():
    models = {
        "SVM_Linear": SVC(
            kernel='linear',
            C=0.1,             # Reduced C to increase regularization and prevent overfitting on small N
            class_weight=custom_weights, # Prioritizes patient detection (Recall)
            probability=True,  # Required for AUC-ROC calculation and threshold tuning
            random_state=42
        ),
        "SVM_RBF": SVC(
            kernel='rbf',
            C=1.0,
            gamma=0.01,        # Fixed gamma for smoother decision boundaries compared to 'scale'
            class_weight=custom_weights,
            probability=True,
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,       # Shallow trees to prevent "memorizing" the noise in small datasets
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50,   # Fewer estimators to maintain model stability with N=108
            learning_rate=0.05,
            max_depth=3,       # Low depth to avoid high variance/overfitting
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=2.5 # XGBoost's parameter to balance classes and boost Recall
        )
    }
    return models
