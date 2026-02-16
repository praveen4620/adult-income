import xgboost as xgb

def get_model():
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)