# numpy-llm

A from-scratch autograd framework and a configurable GPT-style language model,
built to run **controlled training experiments**: which optimizer, activation,
normalization and positional encoding actually work, and *why*, measured with
built-in training telemetry.

Everything that matters is hand-written: the tensor/autograd engine, every
layer, all seven optimizers, the training loop and the telemetry. numpy is
used only for raw array math (and CuPy can replace it to run on a GPU).

| Component | Variants |
|---|---|
| Optimizers | SGD+momentum, AdamW, Lion, Muon, NorMuon, Sophia, Shampoo |
| Activations | ReLU, GELU, SiLU, Squared ReLU, GeGLU, SwiGLU |
| Normalization | LayerNorm, RMSNorm, DyT, ScaleNorm, none |
| Positional | learned absolute, sinusoidal, RoPE |
| LR schedules | constant, warmup+cosine, WSD |

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the engine (autograd vs numerical gradients, optimizer sanity)
python tests/test_gradcheck.py
python tests/test_optimizers.py
python tests/test_metrics.py

# 2. get data (TinyStories validation split, ~20 MB download; also writes the test split)
python scripts/prepare_data.py --tinystories

# 3. sweep learning rates (7 optimizers x 4 LRs, short runs), save the winners
python scripts/lr_sweep.py --write

# 4. run the experiments (resumable: rerun the same command after a crash)
python scripts/run_experiments.py --ids 01-18      # stage 1, 3 seeds each
python scripts/run_experiments.py --ids 19-50,R01-R10

# 5. tables + charts, and the command to confirm the top 5 with extra seeds
python scripts/analyze.py

# 6. final candidates only: held-out test set, leakage checks, human ratings
python scripts/rate_samples.py results/runs/<run_a> results/runs/<run_b>
python scripts/final_eval.py results/runs/<run_a> results/runs/<run_b>

# extras
python scripts/baselines.py                         # n-gram reference models
python scripts/eval_length.py --ids 01,17,18       # positional length generalization
python scripts/generate.py results/runs/01_adamw_gelu_layernorm_learned_s1
```

Open `results/dashboard.html` in a browser at any time. It's rebuilt after every
run, and every run folder has its own `report.html`.

On Windows PowerShell the commands are the same. To start a long batch before
bed, keep the PC from sleeping; if it reboots, rerun the command and finished
runs are skipped.

## The experiment plan

`experiments/configs.json` defines 60 configs (96 runs). Every config changes
fields relative to the **baseline**: AdamW + GELU + LayerNorm + learned
positions, 2 layers, width 64, 4 heads, context 128, 2,000 steps of 32×128
tokens, warmup + cosine schedule, weight decay 0.1, gradient clipping 1.0.

| Group | Configs | What it tests | Seeds |
|---|---|---|---|
| A | 01–18 | One change at a time vs the baseline | 3 |
| B | 19–25 | Every optimizer on the modern stack (SwiGLU + RMSNorm + RoPE) | 1 |
| C | 26–31 | Activation challengers on the modern stack | 1 |
| D | 32–35 | DyT: can the model skip normalization? | 1 |
| E | 36–37 | ScaleNorm | 1 |
| F | 38–40 | No norm: does orthogonalization stabilize training without one? | 1 |
| G | 41–43 | Positional encoding on the modern stack | 1 |
| H | 44–45 | LayerNorm vs RMSNorm in the modern stack | 1 |
| I | 46–50 | Wildcards | 1 |
| R | R01–R10 | Random draw (seed 2026) from the 580 remaining combos | 1 |

After the runs, `analyze.py` prints which top-5 configs have only one seed
and the exact command to rerun them with 3 seeds (~10 more runs). Decide the
winner only after that.

**Decide your success metric before looking at results.** The default is
final validation loss at a fixed token budget (2,000 steps).

### Learning rates

`experiments/lrs.json` holds one LR per optimizer (starting guesses until you
run the sweep). Muon, NorMuon and Shampoo are scaled to share AdamW's LR range;
SGD, Lion and Sophia need their own. Stability-sensitive configs (no norm,
DyT) can get their own LR:

```bash
python scripts/lr_sweep.py --config-id 15 --write   # stored as "config:15"
```

If the sweep warns that the best LR is on the edge of the grid, pass a wider
`--grid` and sweep again.

### Changing the model size

If stage-1 differences are smaller than the seed-to-seed spread, the model is
too small to separate the options. Scale it and use a separate results folder:

```bash
python scripts/run_experiments.py --ids 01-18 --set d_model=128 n_layer=4 --out results_big/runs
```

Parameter count ≈ 12 × n_layer × d_model² + embeddings (baseline: 118,784).

## Metrics and reports

Every run writes `report.html` (open it in a browser) and the runner rebuilds
`results/dashboard.html` after each run: a sortable leaderboard of all configs,
ranking and curve charts, speed and generalization scatter plots, the stage-1
panels, confirmed (3+ seed) configs, human ratings and final evaluations.
`scripts/analyze.py` also exports the charts as PNGs and the tables as CSV.

| When | Metric | How |
|---|---|---|
| Every logged step | training loss, perplexity, learning rate, gradient norm, clipping rate, elapsed time, characters processed, memory | `metrics.jsonl` |
| Every eval | validation loss / top-1 accuracy / BPC / perplexity; the same on fixed training batches; generalization gap | 20 fixed batches, identical for every run |
| End of run | best validation loss and step, BPC, top-1 accuracy, gap, parameters, peak memory, time to each target loss (2.0 … 1.2) | `summary.json` |
| End of run | full validation split (every character), generation samples from 5 fixed prompts, distinct-n, rep-4, real-word rate, story-ending choice accuracy, inference speed | `summary.json` → `final_eval` |
| Final candidates | held-out test loss / BPC / accuracy, story-ending on test, latency and throughput, memorization of training text, val/test overlap with train, human ratings | `scripts/final_eval.py` |

Definitions:
- **BPC** (bits per character) = loss / ln 2. **Perplexity** = e<sup>loss</sup>.
- **Generalization gap** = validation loss − loss on fixed training batches.
- **Time to target**: first evaluation where validation loss ≤ target; train
  time excludes evaluation time.
- **Peak memory**: process resident memory sampled right after backward, when
  the graph is alive (GPU pool memory with CuPy).
- **Story-ending choice**: the model reads ~79 characters of a story and must
  score the real next sentence above 3 sentences from elsewhere (chance 25%).
  The main score is calibrated (PMI: log-likelihood with the context minus
  log-likelihood without it, Holtzman et al. 2021), because raw likelihood
  favors generic sentences; the raw score is reported too. Checked with an
  oracle that knows the text (98–100%) and with shifted labels (0%).
- **Real-word rate**: share of generated words that occur in the training text.
  **distinct-n**: unique word n-grams / all. **rep-4**: share of word 4-grams
  that repeat earlier in the same sample (looping).
- **n-gram baselines**: interpolated Witten–Bell character models of order
  1–8 on the same training text. They're the "other models" every run is
  compared with: a different model family evaluated on the exact same text in
  the same unit. Comparing with GPT-2 or LLaMA numbers wouldn't be
  meaningful: they use different tokenizers and data, so their published
  losses aren't in the same units.
- **Leakage**: share of 32/64/128-character windows of val/test (and of
  generated text) that occur verbatim in the training text.

The **test split** is the text right after the train/val text in the corpus
(`--test-chars`, default 500K). Touch it only with `final_eval.py`, only for
final candidates, and don't change your choice because of it.

Evaluation adds roughly 10–15% to each run. To skip it for a quick run:
`--set final_eval=false train_eval=false`.

The added metrics don't change training: losses and Loss Change Allocation
are bit-identical to runs made before they existed, so old and new runs
compare directly (old runs show "—" for metrics they didn't record).

## Telemetry

Every run writes `metrics.jsonl` with, per logged step:

- **Per layer group** (embed, block0..N, final_norm, head): gradient norm,
  weight norm, update-to-weight ratio ‖ΔW‖/‖W‖, and cumulative **Loss Change
  Allocation** (first-order, Lan et al. 2019): Σ g·Δθ. Negative means that
  group reduced the loss. The first-order version overestimates the total
  loss change, so compare groups to each other, not to the loss curve.
- **FFN:** dead-unit rate (units never active on the batch) and activation sparsity.
- **Attention:** entropy per head (low = sharp/collapsed head).
- **Residual stream:** norm after each block.

## Implementation notes

- **Autograd:** each op stores a closure that receives the output gradient.
  Closures never reference their own output, which avoids reference cycles
  (graphs are freed immediately instead of waiting for Python's cycle collector).
- **Gated FFNs** (GeGLU, SwiGLU) use hidden width ⅔·4d so their parameter
  count matches the plain FFNs.
- **Sinusoidal positions** are scaled to the init scale of a learned position
  table (RMS 0.02). The raw table (RMS ~0.71) drowns out the token embeddings
  at this init and makes sinusoidal look far worse than it is; with the
  scaling, sinusoidal vs learned differs only in fixed vs trained.
- **Muon / NorMuon / Shampoo** apply only to 2D weight matrices inside the
  transformer blocks; embeddings, the output head and gains use AdamW at the
  same LR. Muon updates are scaled by 0.2·√max(m,n) to match AdamW's update
  RMS. NorMuon normalizes per output neuron (a column, since weights are
  stored as in×out). Shampoo grafts its step size from Adam and recomputes
  L^-1/4, R^-1/4 every 10 steps via eigendecomposition with relative damping.
- **Sophia** uses the Gauss–Newton–Bartlett Hessian estimator: every 10 steps
  it samples labels from the model's own predictions, backpropagates, and
  uses B·ĝ² as the diagonal Hessian estimate. Costs one extra
  forward/backward per 10 steps.
- **Data order and init** depend only on the seed, so seed 1 gives every
  config identical starting weights (where shapes match) and identical batches.
  Validation always uses the same fixed batches.

## Running on Kaggle (GPU)

The same code runs on a GPU through CuPy:

```python
!git clone https://github.com/<you>/numpy-llm.git
%cd numpy-llm
!pip install cupy-cuda12x          # skip if already installed
!python scripts/prepare_data.py --tinystories --max-chars 20000000
!NLLM_BACKEND=cupy python scripts/run_experiments.py --ids <winner> --set d_model=384 n_layer=6 n_head=6 max_steps=20000 --out /kaggle/working/final
```

Save outputs under `/kaggle/working/` so they survive the session. For the
final model, re-check the learning rate at an intermediate size first: the
best LR for a tiny model is usually too high for a bigger one.

The CuPy path is written for (every array op goes through `xp`) but has not
been tested on a GPU yet. Run `tests/test_gradcheck.py` with
`NLLM_BACKEND=cupy` on Kaggle first; a mismatch there points to the op to fix.

## Project layout

```
nllm/
  tensor.py       autograd engine and all differentiable ops
  nn.py           layers, norms, activations, attention, GPT
  positional.py   sinusoidal and RoPE tables
  optim.py        the seven optimizers
  schedule.py     LR schedules
  train.py        training loop, telemetry, checkpoints
  data.py         character-level dataset
  experiments.py  config resolution
  metrics.py      evaluation metrics (full-split eval, generation, diversity, benchmark, speed, leakage)
  ngram.py        n-gram baseline models
  report.py       HTML run reports and dashboard
scripts/          prepare_data, lr_sweep, run_experiments, analyze, generate, eval_length,
                  final_eval, rate_samples, baselines
experiments/      configs.json (60 configs), lrs.json
tests/            gradient checks, optimizer tests, metric tests
```

## Honest limits

- Character-level models of ~120K parameters are tiny. Results describe this
  scale; some effects (especially second-order optimizers) are known to change
  with scale, which is worth testing rather than assuming.
- NorMuon, DyT and Sophia are implemented from their papers' descriptions;
  small details may differ from the authors' code.
- This is a careful reproduction and comparison of existing methods, not a
  new method. That's the point: the value is in the rigor and the explanations.
