import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_dataset(path="../data/processed/feature_extracted_dataset.csv"):
    df = pd.read_csv(path)
    return df

def train_model(df):
    print("Preparing features for Linear Regression (Ridge)...")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    # Pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(random_state=42))
    ])

    params = {
        "regressor__alpha": [0.1, 1.0, 10.0, 100.0]
    }

    print("Starting GridSearchCV...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best Hyperparameters:", grid.best_params_)
    print("Best CV Score (MSE):", -grid.best_score_)

    best_model = grid.best_estimator_
    
    print("\nEvaluating on Test Set...")
    test_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    print("Test MSE:", mse)
    print("Test R² :", r2)

    joblib.dump(best_model, "../models/best_linear_regression_model.joblib")
    print("\nModel saved to: models/best_linear_regression_model.joblib")

    return {
        "model": best_model,
        "mse": mse,
        "r2": r2
    }

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print("Training Linear Regression model...")
    train_model(df)
