
# Data Report (EDA Summary)
## Shape (Row/Column Counts)
* The provided dataset shape is **664 rows** and **39 columns**. *(Note: The attached output provides a single shape and does not explicitly break down the counts between train and test sets).*
## Class Distribution (`start_category`)
The target variable exhibits class imbalance, leaning heavily towards the YELLOW and GREEN categories:
* **YELLOW**: 284 counts (42.77%)
* **GREEN**: 176 counts (26.51%)
* **RED**: 135 counts (20.33%)
* **BLACK**: 69 counts (10.39%)
## Missing Values
Several features contain missing data, sorted by highest frequency:
* **pain_score**: 237
* **dbp**: 142
* **sbp**: 141
* **pain_assessable**: 140
* **o2sat**: 119
* **vs_sbp_min**: 83
* **vs_sbp_max**: 83
* **vs_sbp_mean**: 83
* **temperature**: 82
* **heartrate**: 80
* **vs_o2sat_min**: 71
* **vs_o2sat_max**: 71
* **vs_o2sat_mean**: 71
* **resprate**: 62
* **vs_temperature_max**: 37
* **vs_heartrate_min**: 31
* **vs_heartrate_max**: 31
* **vs_heartrate_mean**: 31
* **vs_resprate_min**: 17
* **vs_resprate_max**: 17
* **vs_resprate_mean**: 17
## Feature Types
**Numeric (32 features):**
`subject_id`, `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`, `bp_unobtainable`, `pain_score`, `pain_assessable`, `gcs_eye`, `gcs_verbal`, `gcs_motor`, `gcs_total`, `avpu_ordinal`, `follows_commands`, `vs_heartrate_min`, `vs_heartrate_max`, `vs_heartrate_mean`, `vs_resprate_min`, `vs_resprate_max`, `vs_resprate_mean`, `vs_sbp_min`, `vs_sbp_max`, `vs_sbp_mean`, `vs_o2sat_min`, `vs_o2sat_max`, `vs_o2sat_mean`, `vs_temperature_max`, `n_vitalsign_readings`, `n_diagnoses`, `n_home_meds`
**Categorical / Object (7 features):**
`gender`, `race`, `arrival_transport`, `avpu`, `consciousness_source`, `chiefcomplaint`, `start_category` (Target)
## Notable Distributions or Outliers
*The attached EDA output does not contain the summary statistics (mean, std, min, max, percentiles) required to identify distributions or specific outliers.*
## Data Quality Concerns
* **High Missingness in Critical Vitals:** `pain_score` is missing for 35.7% of the dataset (237/664 rows). Vital signs such as `dbp` and `sbp` are missing in over 20% of the rows (142 and 141, respectively).
* **Correlated Missingness Patterns:** There are identical missing counts for related aggregated variables (e.g., 83 missing for all `vs_sbp_*` columns, 71 missing for all `vs_o2sat_*` columns, 31 missing for all `vs_heartrate_*` columns). This indicates that when a vital sign is missing, its aggregated history is also entirely missing for that patient. 
* **Imputation Required:** Given the volume of missing data, dropping rows with missing values would result in severe data loss, making rigorous imputation strategies mandatory.