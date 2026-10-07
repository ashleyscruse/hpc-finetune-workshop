# Fine-Tune It, Then Make It Faster

A two-session, hands-on workshop on a GPU supercomputer. Participants fine-tune a small
open language model to sort real consumer complaints into six categories, measure it
before and after, and then spread the same training run across 1, 2, and 4 GPU nodes to
see how much faster it gets and why it is never perfectly faster.

No machine learning background is assumed. Everything runs from two Jupyter notebooks.

| Notebook | Session | Length |
|---|---|---|
| `notebooks/01-fine-tune.ipynb` | Teach a model to read complaints (one GPU) | about 100 min |
| `notebooks/02-make-it-faster.ipynb` | Make it faster (four nodes, teams of three to four) | about 75 min |

## What participants do

**Session 1.** Read real complaints, score an untrained model (about 17%, which is
chance for six categories), fine-tune it with LoRA on a few thousand examples, score it
again, test it on complaints they write themselves, and run one controlled experiment
(more data, shorter input, or more passes).

**Session 2.** Predict, then measure, the training time on 1, 2, and 4 nodes. Compute
speedup and efficiency, chart them, and explain the gap from perfect scaling.

## The data

The [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
a US federal public-domain dataset of roughly 10 million complaints. `scripts/stage.py`
streams the bulk CSV once and keeps a balanced sample: 20,000 training and 500 test
complaints for each category.

| Category | CFPB products it covers |
|---|---|
| Credit reporting | Credit reporting, credit repair, consumer reports |
| Debt collection | Debt collection |
| Mortgage | Mortgage |
| Credit card | Credit card, credit card or prepaid card |
| Bank account | Checking or savings account, bank account or service |
| Money transfer & payments | Money transfer, virtual currency, money service, prepaid card |

## The model

[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (Apache 2.0) with a
six-way classification head, trained with LoRA so that well under 1% of the weights
change. One script, `scripts/train.py`, runs on one GPU directly or on many through
`torchrun`.

## Repository

```
notebooks/   the two participant notebooks
scripts/
  setup_shared.sh   instructor: build the shared environment, stage data and model
  stage.py          download the complaints, sample them, download the model
  train.py          fine-tune and score (one GPU or many)
  scale.sh          run train.py on the first N nodes of the current session
  predict.py        classify complaints you write
  peek.py           show category counts and sample complaints
  plot_scaling.py   chart time and speedup
  labels.py         the six categories and the product mapping
launch/      job definitions for a 1-node and a 4-node Jupyter session
```

## Running it yourself (instructors)

Written for a cluster where each GPU node has one GPU and Slurm schedules the jobs (the
notebooks were built on NVIDIA Grace-Hopper nodes).

1. Clone this repository somewhere every participant can read.
2. From a GPU compute-node session, build the shared environment and stage everything:

   ```bash
   bash scripts/setup_shared.sh /path/to/shared
   ```

   This builds a Python environment, downloads and samples the data, downloads the
   model, runs a 600-example training check, and opens read access.
3. Set `SHARED` in the first code cell of both notebooks to that path.
4. Participants clone the repository into their own space, start a Jupyter session
   (1 node for Session 1, 4 nodes for Session 2), and open the notebook.

Session 2 starts its multi-node runs from inside the Jupyter session with `srun`, so
the session must hold at least as many nodes as the largest run.

## Data and model licenses

Complaint data: US federal public domain. Model: Apache 2.0.
