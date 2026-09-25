# Predicting Hostel/PG Tenant Churn Using Micro-Complaint Patterns

**Name:** Bhuvaneshwari D
**Roll Number:** CB.SC.U4CSE23612
**Class:** CSE-G

---

## Problem Statement

Hostel, PG, and co-living operators typically lose tenants abruptly at the end of a lease cycle, with little warning beyond a final move-out notice. Tenant dissatisfaction, however, rarely emerges suddenly — it accumulates through a series of small, individually tolerable complaints (inconsistent WiFi, delayed maintenance, food quality issues, noise, poor hygiene) that a tenant may never formally report, but which collectively erode their willingness to renew.

## Objectives

- Predict tenant renewal vs. churn risk from the self-reported **frequency** and **severity** of minor grievances over a stay.
- Capture whether each grievance was **formally reported** or **silently tolerated**, since silent dissatisfaction is invisible to standard ticketing systems.
- Build a classification model that flags properties or tenant segments at high churn risk from accumulated micro-complaint patterns, rather than waiting for exit interviews.
- Surface actionable, plain-English business recommendations for PG/co-living operators.

## Data Collection Method

- **Primary source:** A structured Google Forms survey distributed to current and recent tenants of hostels, PGs, and co-living spaces across multiple cities (Bangalore, Coimbatore, Mysore, Chennai, Hyderabad, Vijayawada, Kolkata).
- **Real responses collected:** 44 (42 forming the valid, de-duplicated analysis cohort).
- **Instrument:** Respondent context (accommodation type, stay duration, rent range, city, gender), an 8-category micro-complaint grid (WiFi, maintenance, food, noise, cleanliness, water, security, management communication — each rated for frequency and reporting behavior), and outcome variables (renewal likelihood, NPS score).
- **Augmentation:** Because 44 real responses is small for stable model training, the dataset was extended to 480 rows (436 synthetic) using a documented, non-generative **bootstrap resampling with class-aware probabilistic perturbation** — every synthetic row is flagged via `is_synthetic` and linked to its real parent respondent via `group_id`. Full methodology is in `analysis.ipynb` and the report.
- **Secondary/validation source:** 99 publicly available tenant reviews for a PG/co-living operator, scraped from [Trustpilot](https://www.trustpilot.com/review/zolostays.com), keyword-tagged against the same 8 complaint categories and used only as an external validation check — never merged into the training data.
- No personally identifying information (names, emails, phone numbers) is present in any collected, cleaned, or augmented file.

## Analytics Methods

- **Feature engineering:** total complaint burden (aggregate severity across all 8 categories), silent tolerance ratio (share of complaints never formally reported), and a binary churn label (`renewal_likelihood ≤ 2` = churn).
- **Models trained:** Logistic Regression (interpretable baseline), Random Forest, and Gradient Boosting.
- **Methodology correction:** An initial run showed implausibly high accuracy (up to 98.96%) caused by group leakage — synthetic "sibling" rows of the same real respondent appearing in both train and test sets — compounded by class imbalance in the real churn rate (22.73%). This was corrected using a **GroupShuffleSplit** keyed on `group_id` (so no respondent's rows cross the train/test boundary) and **inverse class-frequency weighted sampling** during augmentation. All reported results use this corrected, leak-free split.
- **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance/coefficients.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (best)** | 0.755 | 0.767 | 0.688 | 0.725 | **0.866** |
| Random Forest | 0.667 | 0.652 | 0.625 | 0.638 | 0.846 |
| Gradient Boosting | 0.716 | 0.686 | 0.729 | 0.707 | 0.848 |

**Top predictive features:** stay duration, reporting behavior, silent tolerance ratio, total complaint burden, management communication frequency.

**Statistically significant drivers of non-renewal (Spearman correlation, α = 0.05):** Cleanliness & Hygiene (ρ = −0.396, p = 0.008) and Management Communication (ρ = −0.311, p = 0.040).

**Key business insights:**
1. Roughly 11–14% of tenants who experience a given issue never formally report it — a blind spot invisible to ticket-volume-based monitoring.
2. Management communication is the strongest actionable lever, with a statistically significant link to non-renewal.
3. Cleanliness & hygiene lapses are tolerated far less than WiFi or food issues — they are a deal-breaker category, not an annoyance.
4. Tenure and reporting style predict churn better than any single complaint category.
5. Independent public review data (Trustpilot) corroborates the survey's complaint-category ranking.

**Recommendations:** fast guaranteed response SLAs for water/hygiene/maintenance tickets, periodic pulse check-ins independent of the ticketing system, a documented housekeeping audit cadence, and an early-warning churn score built from the model's top features, applied roughly a month ahead of lease renewal.

## Repository Structure

```
README.md                    — this file
data/
  cleaned_dataset.csv         — cleaned real survey responses (PII removed)
  augmented_dataset_v2.csv    — leak-free, class-balanced augmented dataset (is_synthetic + group_id)
analysis.ipynb                — full notebook: preprocessing, EDA, augmentation,
                                 feature engineering, modelling, evaluation, business insights
Case_Study_Report.pdf         — full written report
```

## References

1. Ahmad, A.K., Jafar, A. and Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning. *Journal of Big Data*, 6(28).
2. Spanoudes, P. and Nguyen, T. (2017). *Deep Learning in Customer Churn Prediction: Unsupervised Feature Learning on Abstract Company Independent Feature Vectors*. arXiv preprint.
3. A questionnaire- and sentiment-analysis-based churn prediction study for the hospitality/PG sector (referenced for comparative context in the Analytics Method section of the report).
4. Trustpilot tenant reviews, Zolostays: https://www.trustpilot.com/review/zolostays.com — used as external validation data only.
