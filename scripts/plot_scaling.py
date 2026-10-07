"""Chart training time and speedup from the scale-*node runs.

    python scripts/plot_scaling.py $SCRATCH/inspire/scale-1node $SCRATCH/inspire/scale-2node ...
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    dirs = [pathlib.Path(d) for d in sys.argv[1:-1]]
    png = pathlib.Path(sys.argv[-1])
    runs = sorted((json.loads((d / "results.json").read_text()) for d in dirs
                   if (d / "results.json").exists()), key=lambda r: r["gpus"])
    if not runs:
        sys.exit("no results.json found yet")
    gpus = [r["gpus"] for r in runs]
    secs = [r["train_seconds"] for r in runs]
    base = secs[0] * gpus[0]
    speedup = [base / s for s in secs]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.bar([str(g) for g in gpus], secs, color="#B30022")
    a.set(title="Training time", xlabel="GPUs (nodes)", ylabel="seconds")
    for x, s in zip(range(len(secs)), secs):
        a.text(x, s, f"{s:.0f}s", ha="center", va="bottom")
    b.plot(gpus, gpus, "--", color="#999999", label="perfect")
    b.plot(gpus, speedup, "o-", color="#B30022", label="ours")
    b.set(title="Speedup", xlabel="GPUs (nodes)", ylabel="times faster than 1 GPU",
          xticks=gpus)
    b.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
