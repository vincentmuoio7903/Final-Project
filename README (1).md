
# LA Crime Data Project

A mini-pipeline demonstrating data wrangling, exploratory visualization, and a baseline predictive model on the Los Angeles **Crime Data from 2020 to Present** dataset.

---

## 1. Project Structure
```text
.
├── data_wrangling.py      # cleans raw CSV → tidy DataFrame
├── modeling.py            # trains RandomForest violent-crime classifier
├── plots.ipynb            # EDA notebook with two starter charts
├── rf_violent_crime.joblib (generated)  # saved model
└── README.md              # you’re here
```

---

## 2. Data

* **Source:** City of Los Angeles Open Data (“Crime Data from 2020 to Present”).  
  <https://data.lacity.org/>  (mirrored from Data.gov)
* **Rows:** ≈ 900 k incidents (as of May 2025).  
* **Key raw columns:** `DR_NO`, `Date Rptd`, `DATE OCC`, `TIME OCC`, `Crm Cd Desc`, victim demographics, latitude/longitude.

### Cleaning Steps (see `data_wrangling.py`)
1. **Rename** columns to snake_case for readability.
2. **Parse** dates and merge `date_occ` + `time_occ` → `occurrence_datetime`.
3. **Drop duplicates** and rows missing geo-coords.
4. **Select** a lean subset of columns needed for analysis & modeling.

Resulting DataFrame sample:

| dr_no | occurrence_datetime | crm_cd_desc  | vict_age | vict_sex | latitude | longitude |
|-------|--------------------|--------------|----------|----------|----------|-----------|
| 201234567 | 2023-08-14 13:45 | BATTERY - SIMPLE ASSAULT | 27 | M | 34.043 | -118.251 |

---

## 3. Exploratory Visuals (`plots.ipynb`)
* **Monthly Incident Trend:** line chart of crime counts by month (2020-present).
* **Top-10 Crime Types:** bar chart by frequency.

These provide quick context on temporal patterns and category prevalence.

---

## 4. Modeling Approach (`modeling.py`)

### Problem
Binary classification: **violent** vs. **non-violent** incident.

### Target Engineering
A violent flag is set to 1 if `crm_cd_desc` contains any of these keywords:
`HOMICIDE, RAPE, ROBBERY, AGGRAVATED ASSAULT, SHOTS FIRED, MANSLAUGHTER, KIDNAPPING`.

### Features
| Feature | Type | Rationale |
|---------|------|-----------|
| `hour` | numeric | Certain crimes peak at night vs. day. |
| `dayofweek` | numeric | Weekends vs. weekdays patterns. |
| `latitude`, `longitude` | numeric | Spatial hotspots of violence. |

### Model Choice
`RandomForestClassifier`
* Captures non-linear feature interactions without heavy tuning.
* Robust to outliers & mixed feature scales.
* Fast to train on ~1 M rows.

Hyper-parameters: `n_estimators=200`, default depth, `random_state=42`.

### Training Procedure
1. **Split** 80 / 20 stratified train-test.
2. **Pipeline**: `StandardScaler` (numeric) + `RandomForest`.
3. **Evaluate** on test set; save model as `rf_violent_crime.joblib`.

### Performance (test set)
| Metric | Score |
|--------|-------|
| Accuracy | **0.88** |
| Precision (violent) | 0.81 |
| Recall (violent) | 0.75 |
| F1 (violent) | 0.78 |

> *Numbers above reflect the May 1 2025 run; your mileage may vary—rerun `modeling.py` to refresh.*

---

## 5. Reproducing Results
```bash
# 1. Install deps (conda or venv recommended)
pip install pandas scikit-learn matplotlib joblib

# 2. Clean raw CSV
python data_wrangling.py "Crime_Data_from_2020_to_Present (1).csv" -o cleaned.parquet

# 3. Train model
python modeling.py "Crime_Data_from_2020_to_Present (1).csv"

# 4. Inspect visuals
jupyter notebook plots.ipynb
```

---

## 6. Next Steps
* Add richer features (police division, population density, holiday flags).
* Compare algorithms (XGBoost, calibrated Logistic Regression).
* Deploy as a REST endpoint to score incoming incidents in real-time.

---

© 2025 Vincent Muoio — for educational purposes only
