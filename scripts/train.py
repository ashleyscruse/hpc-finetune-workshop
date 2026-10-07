"""Fine-tune a small language model to sort consumer complaints into six categories.

One script for both sessions:

    # one GPU (Session 1)
    python scripts/train.py --shared $SHARED --out $SCRATCH/inspire/run1

    # score the untouched base model, no training
    python scripts/train.py --shared $SHARED --out $SCRATCH/inspire/base --epochs 0

    # many GPUs (Session 2): launched by scripts/scale.sh, which runs torchrun per node
    bash scripts/scale.sh 4 --limit 60000

Training uses LoRA: the base model's weights stay frozen and a small set of new weights
(well under 1% of the model) is learned on top of them. Results land in
<out>/results.json, and the trained weights in <out>/adapter when --save-adapter is set.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from labels import LABEL2ID, LABELS  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is staged; never reach the internet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True, help="folder holding data/ and models/")
    ap.add_argument("--model", default="Qwen2.5-0.5B", help="folder name under <shared>/models")
    ap.add_argument("--out", required=True, help="where results.json (and the adapter) go")
    ap.add_argument("--limit", type=int, default=6000, help="training examples to use")
    ap.add_argument("--epochs", type=int, default=1, help="0 = score only, no training")
    ap.add_argument("--batch", type=int, default=32, help="examples per GPU per step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=256, help="tokens read per complaint")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-adapter", action="store_true")
    return ap.parse_args()


def setup_distributed():
    """Return (rank, world_size, device). Works with or without torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    else:
        rank = 0
    return rank, world, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def tokenize(tok, texts, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length",
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def evaluate(model, tok, test, args, device):
    model.eval()
    ids, mask = tokenize(tok, test["text"], args.max_len)
    y = torch.tensor(test["label"].map(LABEL2ID).values)
    preds = []
    for i in range(0, len(ids), 128):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids=ids[i:i + 128].to(device),
                           attention_mask=mask[i:i + 128].to(device)).logits
        preds.append(logits.argmax(-1).cpu())
    preds = torch.cat(preds)
    per_class = {name: round((preds[y == i] == i).float().mean().item(), 4)
                 for i, name in enumerate(LABELS)}
    return round((preds == y).float().mean().item(), 4), per_class


def main():
    args = parse_args()
    rank, world, device = setup_distributed()
    say = print if rank == 0 else (lambda *a, **k: None)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    shared = pathlib.Path(args.shared)
    model_dir = shared / "models" / args.model
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    say(f"GPUs: {world}   trainable weights: {trainable:,} of {total:,} "
        f"({100 * trainable / total:.2f}%)")

    test = pd.read_parquet(shared / "data" / "test.parquet")
    results = {"gpus": world, "nodes": world, "limit": args.limit, "epochs": args.epochs,
               "batch_per_gpu": args.batch, "max_len": args.max_len}

    if args.epochs > 0:
        train = pd.read_parquet(shared / "data" / "train.parquet").head(args.limit)
        ids, mask = tokenize(tok, train["text"], args.max_len)
        y = torch.tensor(train["label"].map(LABEL2ID).values)
        data = TensorDataset(ids, mask, y)
        sampler = DistributedSampler(data, shuffle=True, seed=0) if world > 1 else None
        loader = DataLoader(data, batch_size=args.batch, sampler=sampler,
                            shuffle=sampler is None, drop_last=True)
        net = (DDP(model, device_ids=[device.index or 0] if device.type == "cuda" else None)
               if world > 1 else model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

        steps = len(loader) * args.epochs
        say(f"training on {len(train):,} complaints: {steps} steps of "
            f"{args.batch * world} complaints each ({args.batch} per GPU x {world} GPUs)")
        if world > 1:
            dist.barrier()
        sync(device)
        start = time.perf_counter()
        step = 0
        net.train()
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b_ids, b_mask, b_y in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = net(input_ids=b_ids.to(device), attention_mask=b_mask.to(device),
                               labels=b_y.to(device)).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == steps:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / step * (steps - step)
                    say(f"  step {step:>5}/{steps}   loss {loss.item():.3f}   "
                        f"{elapsed:6.1f}s elapsed   ~{eta:5.0f}s left", flush=True)
        sync(device)
        seconds = time.perf_counter() - start
        results.update(train_examples=len(train), steps=steps,
                       train_seconds=round(seconds, 2),
                       examples_per_second=round(steps * args.batch * world / seconds, 1))
        say(f"training took {seconds:.1f}s "
            f"({results['examples_per_second']:,.0f} complaints per second)")

    if rank == 0:
        acc, per_class = evaluate(model, tok, test, args, device)
        results.update(accuracy=acc, per_class=per_class, test_examples=len(test))
        print(f"accuracy on {len(test):,} complaints it never saw: {100 * acc:.1f}%")
        (out / "results.json").write_text(json.dumps(results, indent=2))
        if args.save_adapter and args.epochs > 0:
            model.save_pretrained(out / "adapter")
            tok.save_pretrained(out / "adapter")
            print(f"saved the trained model to {out / 'adapter'}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
