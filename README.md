# Home Credit Default Risk

This repository contains data for a **credit default prediction** task.

The goal is to estimate whether a loan applicant will have payment difficulties (default) or repay successfully, and output a probability of default for each client.

## 1) Task Definition

This is a supervised binary classification problem:

- `TARGET = 1`: client will have payment difficulties (default)
- `TARGET = 0`: client will repay the loan

Formally, the model learns a function:

`f(X) -> P(default)`

where `X` includes current application information and historical behavior from multiple related tables.

## 2) Dataset Structure

The dataset is **relational** (multi-table), not a single flat CSV.

### Main application tables

- `data/application_train.csv`
- `data/application_test.csv`

Each row is one loan application. Primary key is `SK_ID_CURR`.

- Train split includes `TARGET`
- Test split excludes `TARGET` (you predict it)

### Auxiliary history tables

- `data/bureau.csv` — previous credits from other institutions
- `data/bureau_balance.csv` — monthly bureau credit records
- `data/previous_application.csv` — previous home credit applications
- `data/POS_CASH_balance.csv` — POS / cash loan monthly status
- `data/installments_payments.csv` — installment repayment history
- `data/credit_card_balance.csv` — monthly credit card behavior

### Additional files

- `data/sample_submission.csv` — submission format reference
- `data/HomeCredit_columns_description.csv` — feature descriptions

### Relational property

One client (`SK_ID_CURR`) can map to many rows in auxiliary tables.

That means modeling requires **aggregation + feature engineering** to build one row per client before training.

## 3) Machine Learning Formulation

Input features are derived from:

- current application
- historical loans
- credit history
- repayment behavior

The output is:

- `P(y = 1)` = probability of default

## 4) Model Strategy

### Baseline model (first implementation)

Use **LightGBM** as a strong baseline because it:

- performs very well on tabular data
- handles missing values natively
- is a common high-performing benchmark for this competition

### Alternative models

- Logistic Regression (sanity baseline)
- Random Forest
- XGBoost / CatBoost

### Advanced stage ideas

- richer aggregated feature sets + LightGBM ensembles
- stacking / blending
- temporal-aware feature design

## 5) Model Input and Output

### Raw input

Mixed feature types:

- numerical (income, credit amount, annuity, etc.)
- categorical (gender, occupation type, contract type, etc.)
- temporal (many `DAYS_*` features)

### Processed model input

After preprocessing and aggregation:

- one feature matrix
- one row per `SK_ID_CURR`
- columns from application + aggregated history behavior

Example engineered features:

- average previous loan amount
- count of prior applications
- average installment delay
- credit utilization statistics

### Prediction output

Typical inference call:

`y_pred = model.predict_proba(X)[:, 1]`

This produces probabilities in `[0, 1]`.

## 6) Evaluation Metric

Primary metric: **ROC-AUC**.

Why AUC:

- class distribution is imbalanced (defaults are a minority)
- ranking quality matters more than plain accuracy

## 7) Core Challenges

1. **Multi-table aggregation complexity**
   - careful `groupby` design and statistical summaries are critical
2. **Data leakage risk**
   - temporal columns (`DAYS_*`) and joins can leak future information if used incorrectly
3. **Class imbalance**
   - defaults are relatively rare; evaluate with AUC and consider class weighting
4. **Missing data**
   - many columns have substantial missing rates
5. **Feature engineering dominates performance**
   - aggregation and validation strategy often matter more than model swaps

## 8) End-to-End Pipeline

1. Load raw CSV files
2. Aggregate auxiliary tables to client-level features
3. Join features into a single train/test matrix
4. Preprocess (missing values, encoding)
5. Train LightGBM baseline
6. Predict default probability on test clients
7. Evaluate via ROC-AUC (validation)

## 9) Minimal First Goal

Before full multi-table research, start with a stable baseline:

1. Use only `application_train.csv` and `application_test.csv`
2. Train a LightGBM model
3. Establish a reproducible validation ROC-AUC
4. Then incrementally add aggregated features from auxiliary tables

This staged approach makes debugging, iteration, and performance attribution much easier.

## 10) Suggested Next Steps

- create a baseline notebook/script for application-only training
- add one auxiliary table at a time (for example `bureau.csv` first)
- track feature additions and validation AUC deltas
- keep a leakage-safe validation strategy throughout

