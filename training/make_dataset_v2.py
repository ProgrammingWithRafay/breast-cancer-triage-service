from pathlib import Path
import sys

import pandas as pd

path = Path("data/raw/breast_cancer.csv")
df = pd.read_csv(path)

# Guard: only run on the original 569-row export, so a second run cannot drop more rows.
if len(df) != 569:
    sys.exit(f"Expected 569 rows (v1), found {len(df)}. Restore v1 first.")

# Simulated corrected export from the lab: 30 rows removed.
corrected = df.drop(index=range(100, 130))
corrected.to_csv(path, index=False)

print(f"Rows: {len(df)} -> {len(corrected)}")
print(corrected["target"].value_counts().sort_index())