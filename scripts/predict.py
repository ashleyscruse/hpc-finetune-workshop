"""Ask a trained model which category a complaint belongs to.

    python scripts/predict.py --shared $SHARED --adapter $SCRATCH/inspire/run1/adapter \
        --text "I sent money to my cousin through an app and it never arrived."
"""

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from labels import LABELS  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--model", default="Qwen2.5-0.5B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--text", required=True, action="append", help="repeat for several")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForSequenceClassification.from_pretrained(
        pathlib.Path(args.shared) / "models" / args.model, num_labels=len(LABELS))
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()

    for text in args.text:
        enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = model(**enc).logits.float().softmax(-1)[0]
        top = probs.argsort(descending=True)[:3]
        print(f"\n{text[:100]}")
        for i in top:
            print(f"   {LABELS[i]:<28} {100 * probs[i]:5.1f}%")


if __name__ == "__main__":
    main()
