"""Stage the data and the model once, in a shared folder every participant can read.

Run by the instructor before the workshop, inside a compute-node session (not on a
login node; the sampling pass needs the memory):

    python scripts/stage.py --shared /path/to/shared

Writes:
    <shared>/data/train.parquet   20,000 complaints per category, shuffled
    <shared>/data/test.parquet       500 complaints per category, never trained on
    <shared>/models/<model name>/  the base model, so no participant downloads it
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from labels import LABELS, product_to_label  # noqa: E402

# CFPB stopped publishing complaint narratives in its bulk download (2026), so we use a
# CC0 mirror of its earlier export: only the complaints that have narrative text.
DATA_REPO = "BEE-spoke-data/consumer-finance-complaints"
DATA_FILES = [f"has-text/train-0000{i}-of-00003.parquet" for i in range(3)]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
TRAIN_PER_CLASS = 20_000
TEST_PER_CLASS = 500
MAX_CHARS = 2_000  # narratives longer than this are cut; the model only reads ~256 tokens


def fetch_parquet(raw: pathlib.Path) -> list[pathlib.Path]:
    from huggingface_hub import hf_hub_download

    raw.mkdir(parents=True, exist_ok=True)
    return [
        pathlib.Path(hf_hub_download(DATA_REPO, f, repo_type="dataset", local_dir=raw))
        for f in DATA_FILES
    ]


def sample_balanced(paths: list[pathlib.Path], per_class: int, seed: int) -> pd.DataFrame:
    """Read the files one at a time and keep a uniform random sample of `per_class` rows per category.

    Each row gets a random priority; per category we keep the lowest `per_class`
    priorities seen so far. That is a uniform sample without holding every complaint at once.
    """
    rng = np.random.default_rng(seed)
    kept = {name: pd.DataFrame() for name in LABELS}
    cols = ["Product", "Consumer complaint narrative"]
    for i, path in enumerate(paths):
        chunk = pd.read_parquet(path, columns=cols)
        chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        chunk["label"] = chunk["Product"].map(product_to_label)
        chunk = chunk.dropna(subset=["label"])
        chunk["text"] = chunk["Consumer complaint narrative"].str.slice(0, MAX_CHARS)
        chunk["r"] = rng.random(len(chunk))
        for name, part in chunk.groupby("label"):
            both = pd.concat([kept[name], part[["text", "label", "r"]]])
            kept[name] = both.nsmallest(per_class, "r")
        print(f"  chunk {i:03d}  " + "  ".join(f"{k[:12]}={len(v):,}" for k, v in kept.items()))
    for name, df in kept.items():
        if len(df) < per_class:
            print(f"WARNING: only {len(df):,} narratives for {name!r} (wanted {per_class:,})")
    return pd.concat(kept.values())[["text", "label"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True, help="shared folder participants can read")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    shared = pathlib.Path(args.shared).expanduser()
    data = shared / "data"
    data.mkdir(parents=True, exist_ok=True)

    paths = fetch_parquet(shared / "raw")
    df = sample_balanced(paths, TRAIN_PER_CLASS + TEST_PER_CLASS, args.seed)

    test = df.groupby("label", group_keys=False).apply(lambda g: g.head(TEST_PER_CLASS))
    train = df.drop(test.index)
    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test = test.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train.to_parquet(data / "train.parquet", index=False)
    test.to_parquet(data / "test.parquet", index=False)
    print(f"train {len(train):,} rows, test {len(test):,} rows -> {data}")
    print(train["label"].value_counts().to_string())

    from huggingface_hub import snapshot_download

    model_dir = shared / "models" / args.model.split("/")[-1]
    snapshot_download(args.model, local_dir=model_dir)
    print(f"model -> {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
