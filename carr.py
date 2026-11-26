"""Clean used-car data and train a linear-regression price model.

Expected input: data.csv with a numeric ``Price`` target (typically in lakh).
The test-data.csv sample in this repository has no target column, so it cannot
be used for supervised training by itself.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("data.csv")
TARGET = "Price"


def numeric_value(value):
    """Extract the first numeric value from strings such as '998 CC' or '18 kmpl'."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def price_in_lakh(value):
    """Convert New_Price values to lakh while preserving missing values."""
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    amount = numeric_value(text)
    if pd.isna(amount):
        return np.nan
    if "cr" in text:
        return amount * 100
    return amount


def clean_data(frame):
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()

    # Remove CSV index columns, country fields (if present), and exact duplicates.
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
    country_columns = [
        column for column in frame.columns
        if column.strip().lower() in {"country", "country_name", "country_code"}
    ]
    frame = frame.drop(columns=country_columns, errors="ignore")
    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(frame)}")

    # Convert date/time columns to year-only features; time-of-day is not retained.
    for column in list(frame.columns):
        if "date" not in column.lower():
            continue
        dates = pd.to_datetime(frame[column], errors="coerce")
        if dates.notna().any():
            frame[f"{column}_year"] = dates.dt.year
        frame = frame.drop(columns=column)

    # Strip units from common car specifications. Convert New_Price to lakh.
    for column in ("Mileage", "Engine", "Power"):
        if column in frame.columns:
            frame[column] = frame[column].map(numeric_value)
    if "New_Price" in frame.columns:
        frame["New_Price"] = frame["New_Price"].map(price_in_lakh)

    # Ensure commonly numeric fields and the target are numeric even if CSV cells
    # contain text or inconsistent formatting.
    for column in ("Year", "Kilometers_Driven", "Seats", TARGET):
        if column in frame.columns:
            frame[column] = frame[column].map(numeric_value)

    if TARGET not in frame.columns:
        raise ValueError(
            f"Training data must contain a '{TARGET}' column. "
            "The repository's test-data.csv is unlabeled; provide labeled data.csv."
        )

    # Rows without a known target cannot train a supervised regression model.
    missing_target = int(frame[TARGET].isna().sum())
    if missing_target:
        print(f"Rows removed for missing {TARGET}: {missing_target}")
        frame = frame.dropna(subset=[TARGET])

    # Drop features with no observed values; imputers cannot infer them.
    empty_columns = [column for column in frame.columns if column != TARGET and frame[column].isna().all()]
    if empty_columns:
        print(f"All-missing feature columns removed: {empty_columns}")
        frame = frame.drop(columns=empty_columns)

    return frame.reset_index(drop=True)


def report_data_quality(frame):
    print("\nCleaned data preview:")
    print(frame.head())
    print("\nMissing values by column:")
    print(frame.isna().sum().sort_values(ascending=False).to_string())

    # Report IQR outliers for review rather than automatically deleting them.
    # High prices and powerful engines may be valid luxury-car observations.
    numeric = frame.select_dtypes(include=np.number)
    outlier_counts = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if count:
            outlier_counts[column] = count
    print("\nIQR outliers by numeric column (reported, not dropped):")
    print(pd.Series(outlier_counts, dtype="int64").to_string() if outlier_counts else "None detected")
    print("\nNoise checks: unit text was parsed from Mileage/Engine/Power/New_Price; "
          "unparseable values are treated as missing and imputed.")


def train_model(frame):
    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    if len(frame) < 5:
        raise ValueError("At least five labeled rows are required for a train/test split.")

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nLinear regression evaluation (Price units match the input target):")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    return model


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Add labeled training data with a '{TARGET}' column."
        )
    raw_data = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(raw_data)} rows and {len(raw_data.columns)} columns from {DATA_FILE}.")
    cleaned_data = clean_data(raw_data)
    report_data_quality(cleaned_data)
    train_model(cleaned_data)


if __name__ == "__main__":
    main()
