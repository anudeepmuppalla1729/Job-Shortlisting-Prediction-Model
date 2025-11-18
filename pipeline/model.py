import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

def load_dataset(path="../data/processed/feature_extracted_dataset.csv"):
    df = pd.read_csv(path)
    return df


def train_model(df):
    print("Preparing features...")

    feature_cols = [
        "candidate_skill_count",
        "recruiter_skill_count",
        "weighted_match_sum",
        "total_recruiter_weight",
        "weighted_match_ratio",
        "experience_years"
    ]

    X = df[feature_cols]
    y = df["label_score"]

    # Train-Test Split (10% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    # Hyperparameter grid
    params = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10, 20],
        "criterion": ["squared_error", "friedman_mse"]
    }

    print("Starting GridSearchCV (5-fold cross-validation) ...")

    base_model = DecisionTreeRegressor(random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best Hyperparameters:", grid.best_params_)
    print("Best CV Score (MSE):", -grid.best_score_)

    # Train final model using best params
    best_params = grid.best_params_
    final_model = DecisionTreeRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_pred = final_model.predict(X_test)

    print("Test MSE:", mean_squared_error(y_test, test_pred))
    print("Test R² :", r2_score(y_test, test_pred))

    # Save Model
    joblib.dump(final_model, "../models/best_model.joblib")
    print("\nModel saved to: models/best_model.joblib")

    return final_model



def predict_single(model, features: dict):
    """
    features = {
        "candidate_skill_count": int,
        "recruiter_skill_count": int,
        "weighted_match_sum": float,
        "total_recruiter_weight": float,
        "weighted_match_ratio": float,
        "experience_years": int
    }
    """

    df = pd.DataFrame([features])

    score = model.predict(df)[0]
    shortlist = "Yes" if score >= 3.5 else "No"

    return {
        "qualification_score": round(float(score), 2),
        "shortlist": shortlist
    }



if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()

    print("Training model...")
    model = train_model(df)

    example = {
        "candidate_skill_count": 6,
        "recruiter_skill_count": 4,
        "weighted_match_sum": 8,
        "total_recruiter_weight": 14,
        "weighted_match_ratio": 8 / 14,
        "experience_years": 5
    }

    res = predict_single(model, example)
    print("\nExample Prediction:", res)
