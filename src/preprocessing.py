import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

TARGET_COL = "start_category"

# Excluded entirely — outcome leakage (see data-report.md for the check that proved this)
LEAKAGE_COLS = [
    "n_diagnoses",
    "n_vitalsign_readings",
    "vs_heartrate_min", "vs_heartrate_max", "vs_heartrate_mean",
    "vs_resprate_min", "vs_resprate_max", "vs_resprate_mean",
    "vs_sbp_min", "vs_sbp_max", "vs_sbp_mean",
    "vs_o2sat_min", "vs_o2sat_max", "vs_o2sat_mean",
    "vs_temperature_max",
]

# Kept aside for the submission file, not used as a model feature
DROP_COLS = ["subject_id", "avpu"]  # avpu dropped — redundant with avpu_ordinal

ALL_NUMERIC = [
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "pain_score", "gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total",
    "avpu_ordinal", "follows_commands", "n_home_meds",
    # engineered — added by engineer_features(), must be listed here
    "shock_index", "pulse_pressure", "map",
    "flag_resprate_high", "flag_sbp_low", "flag_no_commands",
]

ALL_CATEGORICAL = [
    "gender", "race", "arrival_transport", "consciousness_source",
    "bp_unobtainable", "pain_assessable",
]


def engineer_features(df):
    df = df.copy()
    df["shock_index"] = df["heartrate"] / df["sbp"].replace(0, np.nan)
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3
    df["flag_resprate_high"] = (df["resprate"] > 30).astype("Int64").fillna(0).astype(int)
    df["flag_sbp_low"] = (df["sbp"] < 90).astype("Int64").fillna(0).astype(int)
    df["flag_no_commands"] = (df["follows_commands"] == 0).astype(int)
    return df


def get_feature_cols():
    numeric_cols = [c for c in ALL_NUMERIC if c not in LEAKAGE_COLS + DROP_COLS]
    categorical_cols = [c for c in ALL_CATEGORICAL if c not in LEAKAGE_COLS + DROP_COLS]
    return numeric_cols, categorical_cols


def fit_preprocessor(train_df, numeric_cols, categorical_cols):
    train_df = engineer_features(train_df)

    imputer = IterativeImputer(random_state=42)
    scaler = RobustScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    imputer.fit(train_df[numeric_cols])
    scaler.fit(imputer.transform(train_df[numeric_cols]))
    encoder.fit(train_df[categorical_cols].astype(str))

    return {
        "imputer": imputer, "scaler": scaler, "encoder": encoder,
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
    }


def transform(df, fitted):
    df = engineer_features(df)
    num = fitted["scaler"].transform(fitted["imputer"].transform(df[fitted["numeric_cols"]]))
    cat = fitted["encoder"].transform(df[fitted["categorical_cols"]].astype(str))
    num_df = pd.DataFrame(num, columns=fitted["numeric_cols"], index=df.index)
    cat_df = pd.DataFrame(cat, columns=fitted["encoder"].get_feature_names_out(), index=df.index)
    return pd.concat([num_df, cat_df], axis=1)


if __name__ == "__main__":
    raw_train = pd.read_csv("data/raw/train_ml03.csv")
    raw_test = pd.read_csv("data/raw/test_pred_ml03.csv")

    numeric_cols, categorical_cols = get_feature_cols()
    fitted = fit_preprocessor(raw_train, numeric_cols, categorical_cols)

    test_ids = raw_test["subject_id"].copy()

    train_processed = transform(raw_train, fitted)
    train_processed["target"] = raw_train[TARGET_COL].values

    test_processed = transform(raw_test, fitted)
    test_processed["subject_id"] = test_ids.values

    train_processed.to_csv("data/processed/train_processed.csv", index=False)
    test_processed.to_csv("data/processed/test_processed.csv", index=False)

    print("train_processed:", train_processed.shape)
    print("test_processed:", test_processed.shape)
    print("target distribution:\n", train_processed["target"].value_counts())