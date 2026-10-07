"""Show how many complaints there are per category, plus a few to read.

    python scripts/peek.py --shared $SHARED --n 3
"""

import argparse
import pathlib
import textwrap

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--shared", required=True)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()

train = pd.read_parquet(pathlib.Path(args.shared) / "data" / "train.parquet")
print(f"{len(train):,} complaints ready for training\n")
print(train["label"].value_counts().to_string(), "\n")
for _, row in train.sample(args.n, random_state=args.seed).iterrows():
    print(f"[{row['label']}]")
    print(textwrap.fill(row["text"][:500], 100), "\n")
