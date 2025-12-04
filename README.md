# **Job Candidate Qualification Scoring Using Machine Learning**

### _A Priority-Based Skill Matching ML System_

---

## **Project Overview:**

This project 

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

---builds a **Machine Learning model** that predicts how well a candidate matches a job requirement based on:

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

The raw input consists of candidate-job pairs.

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

### **2. Candidate Skill Count**

Number of total skills the candidate has.

### **3. Recruiter Skill Count**

Number of skills required for the job. Represents job complexity.

### **4. Experience Years**

Converted to numeric.

These become the final model features:

```
[
  weighted_match_ratio,
  candidate_skill_count,
  recruiter_skill_count,
  experience_years,
  weighted_match_sum,
  total_recruiter_weight
]
```

---

## **Target (Label) — Qualification Score (1–5)**

Because real-world recruiter rating data is unavailable, the project generates a **synthetic but realistic label** simulating recruiter behavior with natural randomness.

---

## **Model Used:**

The project employs a **Model Selection Pipeline** that trains and evaluates multiple algorithms to choose the best performer:

- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Linear Regression**
- **Support Vector Regressor (SVR)**

The model with the lowest Mean Squared Error (MSE) on the test set is automatically selected and saved as `best_model.joblib`.

---

## **Training Methodology:**

To ensure generalization:

- Data is cleaned and validated
- Feature engineering generates numeric inputs
- Data is split into Training and Test sets
- Models are trained using **Cross-Validation** (GridSearchCV for hyperparameter tuning where applicable)
- The best model is selected based on test set performance

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

# **Technical Guide & Usage**

This repository contains an end-to-end pipeline that:

- Cleans raw candidate-job pair data
- Extracts interpretable features
- Trains multiple ML models and selects the best one
- Uses the best model to produce qualification scores and shortlist decisions

## **Repository Structure**

- `pipeline/`
  - `clean.py`: Cleaning utilities.
  - `feature_extraction.py`: Feature extraction logic.
  - `train_and_select_best.py`: **Main training script**. Trains multiple models and saves the best one.
  - `decisionTreeTraining.py`, `randomForestTraining.py`, etc.: Individual model training scripts.
- `main.py`: **Inference entrypoint**. Orchestrates cleaning → feature extraction → prediction.
- `data/`
  - `raw/`: Raw CSVs (e.g., `Raw_Data.csv`).
  - `processed/`: Intermediate cleaned and feature-extracted datasets.
- `models/`: Contains the saved `best_model.joblib`.

---

## **Setup**

1.  **Environment**:
    Ensure you have Python installed. It is recommended to create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Dependencies**:
    Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

---

## **How to Run**

### **1. Make Predictions (Inference)**

To score a raw dataset using the pre-trained model:

```bash
python main.py --raw data/raw/Raw_Data.csv --out data/processed/predictions.csv
```

**Arguments:**
- `--raw` / `-r`: Path to raw CSV.
- `--model` / `-m`: Path to trained model (default: `models/best_model.joblib`).
- `--out` / `-o`: Output CSV path.
- `--cleaned` / `-c`: Path to save intermediate cleaned CSV.

### **2. Re-train the Model**

To retrain the models and select the best one:

1.  Ensure you have a dataset with labels (e.g., `data/processed/feature_extracted_dataset.csv`).
2.  Run the training pipeline:

    ```bash
    python pipeline/train_and_select_best.py
    ```

This script will:
- Load the dataset.
- Train Decision Tree, Linear Regression, Random Forest, SVR, and Gradient Boosting models.
- Compare their MSE and R2 scores.
- Save the best model to `models/best_model.joblib`.

---

## **Project Goals:**

- Build a scalable ML-based candidate matching system
- Reduce recruiter workload
- Improve fairness and consistency in hiring
- Enable automated resume/job screening

## **Future Enhancements:**

- Resume text embedding using NLP
- Job description semantic matching
- Soft skill extraction via LLM
- Real recruiter feedback for real labels
- Deployment on cloud (AWS/GCP/Azure)
- API integration with ATS or HR tools

---
