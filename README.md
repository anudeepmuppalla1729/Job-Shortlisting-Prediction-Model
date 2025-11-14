# **Job Candidate Qualification Scoring Using Machine Learning**

### *A Priority-Based Skill Matching ML System*

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

| Field | Example |
| --- | --- |
| candidate_id | c103 |
| job_id | j180 |
| role | financial analyst |
| candidate_skills | sales|support|docker|seo|c|node|crm|git |
| recruiter_skills | git|prototyping|sales|seo |
| recruiter_priority_list | seo|sales|prototyping|git |
| recruiter_skill_weights | {"seo":5,"sales":4,"prototyping":3,"git":2} |
| experience_years | 11 |

The dataset is cleaned and transformed for training.

---

## **Feature Engineering:**

To make the data usable by ML, raw fields are converted into numerical features:

### **1. Weighted Match Score**

Measures how well candidate skills match recruiter’s weighted skills.

weighted_match=∑(candidate has skill×weight)∑(all recruiter weights)weighted\_match = \frac{\sum(\text{candidate has skill} \times \text{weight})}{\sum(\text{all recruiter weights})}

weighted_match=∑(all recruiter weights)∑(candidate has skill×weight)

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
+ 3.5 * weighted\_match
+ 0.1 * candidate\_skill\_count
- 0.05 * recruiter\_skill\_count
+ noise

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
- decision tree regression
- clean end-to-end pipeline

It provides an efficient, scalable, and intelligent alternative to manual candidate evaluation.