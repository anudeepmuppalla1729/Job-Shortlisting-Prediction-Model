import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import os

def test_model():
    model_path = "models/best_model.joblib"
    data_path = "data/processed/feature_extracted_dataset.csv"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Define features and target
    feature_cols = [
        "candidate_skill_count",
        "recruiter_skill_count",
        "weighted_match_sum",
        "total_recruiter_weight",
        "weighted_match_ratio",
        "experience_years",
    ]
    target_col = "label_score"

    # Check if columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns: {missing_cols}")
        return

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return

    X = df[feature_cols]
    y = df[target_col]

    # Predict
    print("Making predictions...")
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print("-" * 30)
    print(f"Model Performance on {data_path}")
    print("-" * 30)
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    test_model()
