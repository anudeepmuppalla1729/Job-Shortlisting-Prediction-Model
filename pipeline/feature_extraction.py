import pandas as pd
import numpy as np
import ast

def extract_features(df):
    # Ensure list/dict are python objects
    df["candidate_skills_list"] = df["candidate_skills_list"].apply(ast.literal_eval)
    df["recruiter_priority_list"] = df["recruiter_priority_list"].apply(ast.literal_eval)
    df["recruiter_skill_weights_dict"] = df["recruiter_skill_weights_dict"].apply(ast.literal_eval)

    # 1. CANDIDATE_SKILL_COUNT
    df["candidate_skill_count"] = df["candidate_skills_list"].apply(len)

    # 2. RECRUITER_SKILL_COUNT
    df["recruiter_skill_count"] = df["recruiter_skill_weights_dict"].apply(len)

    # 3. WEIGHTED_MATCH_SUM
    def compute_weighted_match(row):
        candidate_skills = set(row["candidate_skills_list"])
        recruiter_weights = row["recruiter_skill_weights_dict"]
        return sum(weight for skill, weight in recruiter_weights.items() if skill in candidate_skills)

    df["weighted_match_sum"] = df.apply(compute_weighted_match, axis=1)

    # 4. toatal_recruiter_weight
    df["total_recruiter_weight"] = df["recruiter_skill_weights_dict"].apply(lambda d: sum(d.values()))

    # 5. weighted_match_ratio
    df["weighted_match_ratio"] = df.apply(
        lambda row: row["weighted_match_sum"] / row["total_recruiter_weight"]
        if row["total_recruiter_weight"] > 0 else 0,
        axis=1
    )

    # 6. label_score (Tartget Variable)
    def compute_label_score(row):
        noise = np.random.normal(0, 0.20)
        score = (
            1
            + 3.5 * row["weighted_match_ratio"]
            + 0.1 * row["candidate_skill_count"]
            - 0.05 * row["recruiter_skill_count"]
            + noise
        )
        return round(np.clip(score, 1, 5), 2)
    df["label_score"] = df.apply(compute_label_score, axis=1)

    return df

if __name__ == "__main__":
    df = pd.read_csv("..data/processed/cleaned_dataset.csv")
    df = extract_features(df)

    df.to_csv("../data/processed/feature_extracted_dataset.csv", index=False)
    print("Feature extraction completed and saved to ../processed/feature_extracted_dataset.csv")
    print(df.head())