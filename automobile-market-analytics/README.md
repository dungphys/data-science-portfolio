# Automotive Resale Market Analysis

An end-to-end data science project that explores an automotive resale
market dataset, builds a predictive model for `Selling_Price`, and
answers a set of concrete business questions for a used-car
marketplace/dealer audience.

```Project's Tree
automotive-market-analytics/
├── data/                          # place automotive_dataset.csv here
├── src/
│   ├── config.py                  # paths, schema, constants
│   ├── data/loader.py             # load + schema validate + clean raw data
│   ├── features/build_features.py # feature engineering
│   ├── eda/eda.py                 # summary stats + plots
│   ├── models/                    # train.py, evaluate.py, predict.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── business/business_questions.py  # the 7 business questions
├── scripts/
│   ├── run_eda.py                 # produces reports/figures + eda_summary.txt
│   ├── run_training.py            # trains + saves models/price_model.joblib
│   └── run_business_analysis.py   # produces reports/business_report.md
├── tests/                     # pytest suite (synthetic fixtures, no real data needed)
├── models/                    # trained model artifact + metrics.json (generated)
├── results/                   # figures + markdown report (generated)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

## Setup

### 1. Add the data
Place data at `data/automotive_dataset.csv` with these columns:

`Make, Model, Year, Fuel_Type, Transmission, Engine_Size, Mileage,
Horsepower, Torque, Owners, Accident_History, Service_History, Color,
Body_Type, Drivetrain, Fuel_Efficiency, Location, Selling_Price`

### 2. How to run

#### **2a. Run locally**

```bash
pip install -r requirements.txt
make eda        # -> results/figures/*.png, results/eda_summary.txt
make train       # -> models/price_model.joblib, models/metrics.json
make business    # -> results/business_report.md
make test        # run the pytest suite
```

#### **2b. Run with Docker**

```bash
docker compose build
docker compose run --rm eda
docker compose run --rm train
docker compose run --rm business
docker compose run --rm test
```
Each service mounts `data/`, `models/`, and `results/` as volumes so
generated artifacts land back on your host filesystem.

---

## Methodology

**Data cleaning** (`src/data/loader.py`): validate schema, drop exact
duplicates, coerce numeric columns, drop rows with missing/non-positive
target, filter implausible `Year`/`Mileage` values, trim whitespace in
text fields.

**Feature engineering** (`src/features/build_features.py`):
- `Vehicle_Age` = current year − `Year` (clipped at 0)
- `Mileage_Per_Year` = `Mileage` / `Vehicle_Age` — usage intensity independent of raw age
- `Power_Per_Liter` = `Horsepower` / `Engine_Size` — performance-tier proxy
- `Service_History_Score` — ordinal encoding of `Service_History` (No/Partial/Full)

**EDA** (`src/eda/eda.py`): missing-value summary, numeric descriptive
stats, Pearson correlation of numeric features with price, grouped
price comparisons by category, and five saved plots (price
distribution, correlation heatmap, price vs. mileage, price by body
type, price by accident history).

**Modeling** (`src/models/`): a scikit-learn `Pipeline` combining a
`ColumnTransformer` (median-imputation + scaling for numeric features,
most-frequent-imputation + one-hot encoding for categoricals) with a
`RandomForestRegressor`. Chosen over a linear model because
price/mileage/age relationships are non-linear (steep early
depreciation, flattening later) and Random Forest handles mixed
numeric/categorical data and feature interactions without manual
transformation, while still exposing interpretable feature
importances. `Model` (specific vehicle model name) is excluded from
the feature set — it's high-cardinality and largely redundant with
`Make`; including it would blow up dimensionality via one-hot encoding
for limited predictive gain.

Data is split 80/20 (train/test, `random_state=42`). Evaluated with
RMSE, MAE, R², and MAPE on the held-out test set.

**Business analysis** (`src/business/business_questions.py`): each
question is answered with a specific, auditable groupby/pivot/
correlation computation over the cleaned + engineered data (methodology
noted in each function's docstring), plus the model's feature
importances for Q1.

---

## Business Questions

| # | Question | How it's answered |
|---|----------|--------------------|
| Q1 | Which vehicle attributes most influence resale price? | Random Forest feature importances (mean decrease in impurity) |
| Q2 | How much does a prior accident reduce resale value? | Group by `Accident_History`, compare mean/median, compute % penalty |
| Q3 | What does the depreciation curve look like as vehicles age? | Median price by `Vehicle_Age` bucket, indexed to the newest bucket |
| Q4 | Which fuel type / body type combinations command the highest resale value? | Median price pivot: `Body_Type` × `Fuel_Type` |
| Q5 | Does a fuller service history translate into measurable resale value? | Median price by `Service_History`, % premium of Full vs. No Service |
| Q6 | Which states show the highest average resale prices? | Mean price by `Location`, states with < 5 listings excluded |
| Q7 | Does higher annualized mileage correlate with lower price? | Pearson correlation of `Mileage_Per_Year` vs. `Selling_Price` |

Run `python scripts/run_business_analysis.py` (after training a model)
to generate `reports/business_report.md` with the actual numbers for
your dataset. The pipeline was smoke-tested end-to-end on a synthetic
sample during development (see `tests/conftest.py` for the fixture
used by the test suite) — on real data, run the three scripts in order
(`run_eda.py` → `run_training.py` → `run_business_analysis.py`) to get
results specific to your `automotive_dataset.csv`.

## Results

This section is meant to be filled in with your dataset's actual
numbers after running `make pipeline` (or the Docker equivalent) —
paste in the contents of `reports/business_report.md` and the metrics
from `models/metrics.json`, or summarize the key findings here for
stakeholders. Suggested structure:

1. **Model performance** — RMSE / MAE / R² / MAPE from `models/metrics.json`, with a sentence on whether it's fit for the intended use (e.g. "±$X typical error, suitable for indicative pricing, not final appraisal").
2. **Top price drivers** — from Q1, translated into plain language.
3. **Answers to Q2–Q7** — one or two sentences per question, pulled from `reports/business_report.md`.
4. **Recommendations** — concrete actions a dealer/marketplace could take (e.g. pricing rules, inventory sourcing, disclosure policy for accident history).

## Testing

```bash
pytest              # or: make test / docker compose run --rm test
```
42 tests cover data cleaning/validation, feature engineering, EDA
summary functions, model training/evaluation/prediction (including a
save/load round-trip), and every business-question function. Tests use
an in-memory synthetic fixture (`tests/conftest.py`) so the suite is
fast, deterministic, and doesn't depend on the real dataset being
present.