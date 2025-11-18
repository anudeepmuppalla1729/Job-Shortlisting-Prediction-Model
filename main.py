import argparse
import os
import pandas as pd
import joblib
import numpy as np

from pipeline.clean import clean_dataset
from pipeline.feature_extraction import extract_features


def predict_single(model, features: dict):
    """Predicts the 'label_score' for a single candidate-job pair and determines if the candidate should be shortlisted."""
    df = pd.DataFrame([features])
    score = model.predict(df)[0]
    shortlist = "Yes" if score >= 3.5 else "No"
    return {"qualification_score": round(float(score), 2), "shortlist": shortlist}



def process_raw_and_predict(raw_input_path: str,
                           model_path: str = "models/best_model.joblib",
                           output_path: str = "data/processed/predictions.csv",
                           cleaned_output_path: str = "data/processed/cleaned_dataset.csv") -> pd.DataFrame:
    """Run cleaning -> feature extraction -> model prediction for a raw data CSV.

    Returns the DataFrame with features + predictions and writes predictions to `output_path`.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load original raw data so we can merge predictions back into it later
    raw_df = pd.read_csv(raw_input_path)

    # 1) Clean raw data (clean_dataset returns the cleaned DataFrame)
    cleaned_df = clean_dataset(raw_input_path, cleaned_output_path)

    # 2) Feature extraction expects string representations for lists/dicts (it uses ast.literal_eval).
    #    Ensure those columns are strings so `extract_features` is robust whether input was list or string.
    df_for_extraction = cleaned_df.copy()
    for col in ["candidate_skills_list", "recruiter_priority_list", "recruiter_skill_weights_dict"]:
        if col in df_for_extraction.columns:
            df_for_extraction[col] = df_for_extraction[col].apply(lambda x: str(x))

    features_df = extract_features(df_for_extraction)

    # 3) Prepare features for model prediction
    feature_cols = [
        "candidate_skill_count",
        "recruiter_skill_count",
        "weighted_match_sum",
        "total_recruiter_weight",
        "weighted_match_ratio",
        "experience_years",
    ]

    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns after extraction: {missing}")

    X = features_df[feature_cols]

    # 4) Load model and predict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    preds = model.predict(X)
    features_df = features_df.copy()
    features_df["predicted_score"] = preds
    features_df["shortlist"] = features_df["predicted_score"].apply(lambda s: "Yes" if s >= 3.5 else "No")

    # 5) Attach predictions back to the ORIGINAL raw dataset 
    pred_cols = ["predicted_score", "shortlist"]

    if ("candidate_id" in raw_df.columns and "job_id" in raw_df.columns
            and "candidate_id" in features_df.columns and "job_id" in features_df.columns):
        merged = raw_df.merge(
            features_df[["candidate_id", "job_id"] + pred_cols],
            on=["candidate_id", "job_id"],
            how="left",
        )

        # Remove any training label column if present in raw data
        merged = merged.drop(columns=["label_score"], errors="ignore")

        merged.to_csv(output_path, index=False)
        print(f"Predictions merged into raw data and saved to: {output_path}")
        print(f"Rows in raw file: {len(merged)} | Rows with predictions: {features_df.shape[0]}\n")
        return merged
    else:
        save_df = features_df.copy()
        if "label_score" in save_df.columns:
            save_df = save_df.drop(columns=["label_score"])
        save_df.to_csv(output_path, index=False)
        print(f"Could not merge with raw data (missing identifiers). Saved features+predictions to: {output_path}")
        print(f"Rows saved: {len(save_df)}\n")
        return save_df


def main():
    parser = argparse.ArgumentParser(description="Run cleaning + feature extraction + prediction on a raw CSV.")
    parser.add_argument("--raw", "-r", default="data/raw/Raw_Data.csv", help="Path to raw CSV input")
    parser.add_argument("--model", "-m", default="models/best_model.joblib", help="Path to trained model joblib file")
    parser.add_argument("--out", "-o", default="data/processed/predictions.csv", help="Output CSV path for predictions")
    parser.add_argument("--cleaned", "-c", default="data/processed/cleaned_dataset.csv", help="Path to save cleaned intermediate CSV")

    args = parser.parse_args()

    result_df = process_raw_and_predict(args.raw, args.model, args.out, args.cleaned)

    print("--- Prediction summary (top 5 rows) ---")
    cols_to_show = [c for c in ["candidate_id", "job_id", "predicted_score", "shortlist"] if c in result_df.columns]
    print(result_df[cols_to_show].head())


if __name__ == "__main__":
    main()