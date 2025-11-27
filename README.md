# **Job Candidate Qualification Scoring Using Machine Learning**

### _A Priority-Based Skill Matching ML System_

---

## **Project Overview:**

This project builds a **Machine Learning model** that predicts how well a candidate matches a job requirement based on:

- candidate skills
- recruiter-required skills
- priority weights assigned by the recruiter
- candidate experience

The model outputs:

- A **qualification score (1–5)**
- An optional **shortlist decision (Yes/No)**

This system can be integrated into:

- Applicant Tracking Systems (ATS)
- HR automation tools
- Resume screening platforms
- Internal hiring dashboards

---

## **Problem Statement:**

Recruiters often manually evaluate job applicants by comparing the candidate's skill set with the required job skills.

This process is:

- slow
- subjective
- inconsistent
- not scalable

The goal is to **automate** this evaluation using a Machine Learning model that:

- considers recruiter-selected skills
- accounts for priority/weightage
- understands candidate skill breadth
- factors in job complexity
- uses real-world experience levels

The output is a **numerical score** that can be used for ranking and shortlisting candidates.

---

## **Input Data:**

The raw input consists of 10,000+ rows of candidate-job pairs:

**Example row:**

| Field                    | Example                                                  |
|--------------------------|-----------------------------------------------------------|
| candidate_id             | c103                                                      |
| job_id                   | j180                                                      |
| role                     | financial analyst                                        |
| candidate_skills         | sales \| support \| docker \| seo \| c \| node \| crm \| git |
| recruiter_skills         | git \| prototyping \| sales \| seo                       |
| recruiter_priority_list  | seo \| sales \| prototyping \| git                       |
| recruiter_skill_weights  | {"seo":5,"sales":4,"prototyping":3,"git":2}              |
| experience_years         | 11                                                        |


The dataset is cleaned and transformed for training.

---

## **Feature Engineering:**

To make the data usable by ML, raw fields are converted into numerical features:

### **1. Weighted Match Score**

Measures how well candidate skills match recruiter’s weighted skills.

Let  $weighted match = 
\frac{\sum(\text{candidate has skill} \times \text{weight})}
{\sum(\text{all recruiter weights})}$.




Range → **0.0 to 1.0**

---

### **2. Candidate Skill Count**

Number of total skills the candidate has.

---

### **3. Recruiter Skill Count**

Number of skills required for the job

Represents job complexity.

---

### **4. Experience Years**

Converted to numeric.

---

These become the final model features:

```
[
  weighted_match_ratio,
  candidate_skill_count,
  recruiter_skill_count,
  experience_years
]

```

---

## **Target (Label) — Qualification Score (1–5)**

Because real-world recruiter rating data is unavailable, the project generates a **synthetic but realistic label** using:

label=1+3.5∗weighted_match+0.1∗candidate_skill_count−0.05∗recruiter_skill_count+noiselabel = 1

- 3.5 \* weighted_match
- 0.1 \* candidate_skill_count

* 0.05 \* recruiter_skill_count

- noise

label=1+3.5∗weighted_match+0.1∗candidate_skill_count−0.05∗recruiter_skill_count+noise

Then clipped between 1 and 5.

This simulates recruiter behavior with natural randomness.

---

## **Model Used:**

The project uses **Decision Tree Regressor**, which is suitable because:

- it works well with low-dimensional structured data
- it naturally captures non-linear relationships
- it is easy to interpret
- it performs well on tabular datasets

Later improvements may include:

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

---

## **Training Methodology:**

To ensure generalization:

- Data is cleaned and validated
- Feature engineering generates numeric inputs
- 90% used for K-Fold Cross Validation
- 10% held out for final evaluation
- Hyperparameter tuning optimizes the tree:

Tuned parameters include:

- max_depth
- min_samples_split
- min_samples_leaf
- criterion

This makes the model stable, accurate, and reliable.

---

## **Model Output:**

For every candidate-job pair, the system returns:

```
{
  "qualification_score": 4.12,
  "shortlist": "Yes"
}

```

Shortlisting logic:

```
score ≥ 3.5 → Yes
score < 3.5 → No

```

---

## **Project Goals:**

- Build a scalable ML-based candidate matching system
- Reduce recruiter workload
- Improve fairness and consistency in hiring
- Enable automated resume/job screening
- Provide ranking for large applicant pools

---

## **Future Enhancements:**

- Resume text embedding using NLP
- Job description semantic matching
- Soft skill extraction via LLM
- Real recruiter feedback for real labels
- Deployment on cloud (AWS/GCP/Azure)
- API integration with ATS or HR tools

---

## **Conclusion:**

This ML project demonstrates a fully automated job-candidate matching system using:

- real-world inspired features
- weighted skill matching logic

# Job Candidate Qualification Scoring (Priority-based skill matching)

This repository contains a small end-to-end pipeline that:

- Cleans raw candidate-job pair data
- Extracts interpretable features (weighted skill matches, counts, experience)
- Uses a trained model to produce a per-row qualification score and shortlist decision

The intention is to provide an easy, local workflow to score large candidate-job datasets and produce a CSV of the original rows augmented with a predicted score and shortlist column.

---

## What this repo contains

- `pipeline/clean.py` — cleaning utilities that normalize skill strings, parse JSON weight maps and create canonical columns.
- `pipeline/feature_extraction.py` — turns cleaned rows into numeric features used by the model.
- `pipeline/model.py` — training utilities (if you want to re-train a model locally).
- `main.py` — the entrypoint: it orchestrates cleaning → feature extraction → prediction and writes the final CSV.
- `data/raw/` — raw CSV(s) you can score (`Raw_Data.csv` is included as example).
- `data/processed/` — cleaned / feature-extracted intermediate datasets and the predictions output.
- `models/best_model.joblib` — pre-trained model used by default for predictions.

---

## Expected input (raw CSV)

The pipeline expects a CSV with at least the following fields so merging back to the original rows works cleanly:

- `candidate_id` (string) — unique id for the candidate
- `job_id` (string) — unique id for the job
- `candidate_skills` (string) — skills separated by `|`, e.g. `python|sql|git`
- `recruiter_skill_weights` (JSON-like string) — e.g. `{"sql":5,"python":4}`
- `recruiter_priority_list` (string) — skills in priority order separated by `|`
- `experience_years` (int)

If your raw CSV contains these columns, the script will merge predictions back into the original rows and write the final CSV with the same columns plus the new columns described below.

If the identifiers (`candidate_id` and `job_id`) are missing, the script will save a features+predictions CSV as a fallback.

---

## Output

By default the pipeline writes to `data/processed/predictions.csv`. When identifiers are present the output will be the original raw rows plus two new columns:

- `predicted_score` — model-predicted qualification score (float; roughly scaled 1–5)
- `shortlist` — `Yes` if `predicted_score >= 3.5`, otherwise `No`

Important: the pipeline will NOT include any ground-truth `label_score` (if present) in the final predictions CSV — only model predictions are written.

---

## How to run (PowerShell example)

1. Activate the virtual environment shipped with the project (or use your Python environment that has the required packages listed in `requirements.txt`):

```powershell
& .\myenv\Scripts\Activate.ps1
```

2. Run the pipeline on the default raw file (this will: clean → extract features → predict → save merged CSV):

```powershell
python .\main.py --raw .\data\raw\Raw_Data.csv --out .\data\processed\predictions.csv
```

3. Check the output file:

```powershell
Get-Content .\data\processed\predictions.csv -TotalCount 10
```

Optional CLI flags:

- `--raw` / `-r`: path to raw CSV (default `data/raw/Raw_Data.csv`)
- `--model` / `-m`: path to the trained model joblib file (default `models/best_model.joblib`)
- `--out` / `-o`: output CSV path (default `data/processed/predictions.csv`)
- `--cleaned` / `-c`: path to save intermediate cleaned CSV (default `data/processed/cleaned_dataset.csv`)

Example with custom model and output:

```powershell
python .\main.py -r .\data\raw\sample_raw_data.csv -m .\models\best_model.joblib -o .\data\processed\my_predictions.csv
```

---

## Notes & troubleshooting

- If the script raises `FileNotFoundError` for the model, check that `models/best_model.joblib` exists or pass `--model` to point to your trained artifact.
- If merging does not include predictions, ensure both `candidate_id` and `job_id` exist in your raw file and are present (and identical) in the cleaned/features DataFrame.
- The feature extraction step expects the cleaned columns for lists/dicts to be string representations (the pipeline already coerces these for robustness). If you modified `pipeline/feature_extraction.py`, be careful with `ast.literal_eval` usage.
- To inspect intermediate outputs, check:
  - `data/processed/cleaned_dataset.csv`
  - `data/processed/feature_extracted_dataset.csv`

Common quick checks:

- Print the first lines of the raw file to verify headers:

```powershell
Import-Csv .\data\raw\Raw_Data.csv | Select-Object -First 5 | Format-Table
```

- If Python raises parsing errors when reading `recruiter_skill_weights`, open a sample row and ensure the string is proper JSON or in the format `{"skill":weight,...}`. `pipeline/clean.py` tries to be tolerant by replacing single quotes with double quotes when safe.

---

## Re-training the model

If you want to re-train the model using the repo's utilities, open `pipeline/model.py`. Typical steps:

1. Generate or update `data/processed/feature_extracted_dataset.csv` by running the cleaning and feature extraction steps.
2. Run the training function in `pipeline/model.py` (or create a small script) to read the features file, train a scikit-learn regressor, and write `models/best_model.joblib`.

If you want, I can add a `train.py` driver that executes a reproducible training run and saves the model, plus a small unit test to validate predictions shape.

---

## Contact / Next steps

- Want the repository packaged into an installable project (with `requirements.txt` and a shorter `run.sh`/`run.ps1`)? I can add that.
- Want a small web API wrapper (Flask/FastAPI) that serves predictions? I can scaffold it and add a simple `curl` example.

That's it — the README now explains what the pipeline does, how to run it, and what output to expect. If you'd like any additions (example sample rows, screenshots, or a small `train.py`), tell me which and I'll add them.
