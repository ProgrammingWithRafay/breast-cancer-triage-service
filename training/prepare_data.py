"""Prepare the raw breast cancer dataset as a CSV file."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

DEFAULT_OUTPUT = Path("data/raw/breast_cancer.csv")
EXPECTED_FEATURES = 30
TARGET_COLUMN = "target"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export the breast cancer dataset to CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def build_dataframe() -> pd.DataFrame:
    """Load the dataset and return features plus target in one DataFrame."""
    dataset = load_breast_cancer()

    # Replace spaces with underscores, e.g. "mean radius" -> "mean_radius"
    columns = [name.replace(" ", "_") for name in dataset.feature_names]

    df = pd.DataFrame(dataset.data, columns=columns)
    # Keep original labels: 0 = malignant, 1 = benign
    df[TARGET_COLUMN] = dataset.target
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation errors (empty if the data is valid)."""
    errors: list[str] = []

    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        errors.append(f"Found {null_count} null value(s) in the dataset.")

    feature_count = df.shape[1] - 1  # exclude the target column
    if feature_count != EXPECTED_FEATURES:
        errors.append(
            f"Expected {EXPECTED_FEATURES} feature columns, found {feature_count}."
        )

    return errors


def main() -> None:
    args = parse_args()
    output_path: Path = args.output

    df = build_dataframe()

    # Stop before writing anything if the data looks wrong
    errors = validate(df)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    # Create the output folder if it does not exist and save the CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Short success summary
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    print(f"Saved dataset to {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Class counts: malignant (0) = {class_counts.get(0, 0)}, "
          f"benign (1) = {class_counts.get(1, 0)}")


if __name__ == "__main__":
    main()
