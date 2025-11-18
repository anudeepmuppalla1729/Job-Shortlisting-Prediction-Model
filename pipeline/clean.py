import pandas as pd
import json

def clean_dataset(input_path, output_path):

    # Load raw data
    
    df = pd.read_csv(input_path)

    # Remove duplicate rows

    df = df.drop_duplicates()


    # Drop rows with missing important values

    df = df.dropna(subset=[
        "candidate_skills", 
        "recruiter_skill_weights",
        "experience_years"
    ])


    # Convert candidate_skills (string → list)

    df["candidate_skills_list"] = (
        df["candidate_skills"].astype(str)
        .apply(lambda x: [s.strip().lower() for s in x.split("|") if s.strip()])
    )


    # Convert recruiter_priority_list (string → list)

    df["recruiter_priority_list"] = (
        df["recruiter_priority_list"].astype(str)
        .apply(lambda x: [s.strip().lower() for s in x.split("|") if s.strip()])
    )

    # Convert recruiter_skill_weights JSON string → dict

    def safe_load_json(x):
        try:
            return json.loads(x.replace("'", '"'))
        except:
            return {}

    df["recruiter_skill_weights_dict"] = (
        df["recruiter_skill_weights"].apply(safe_load_json)
    )

    # Normalize keys
    df["recruiter_skill_weights_dict"] = df["recruiter_skill_weights_dict"].apply(
        lambda d: {k.lower(): int(v) for k, v in d.items()}
    )


    # Standardize role names

    df["role"] = df["role"].astype(str).str.lower().str.strip()

    # Convert experience to int

    df["experience_years"] = df["experience_years"].astype(int)

    # Select ONLY the cleaned columns we want
    cleaned_df = df[[
        "candidate_id",
        "job_id",
        "role",
        "candidate_skills_list",
        "recruiter_priority_list",
        "recruiter_skill_weights_dict",
        "experience_years"
    ]]

    # Save cleaned output
    cleaned_df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to: {output_path}")
    return cleaned_df


if __name__ == "__main__":
    clean_dataset(
        input_path="../data/raw/Raw_Data.csv",
        output_path="../data/processed/cleaned_dataset.csv"
    )
