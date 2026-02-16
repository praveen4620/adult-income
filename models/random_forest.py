from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(random_state=42)