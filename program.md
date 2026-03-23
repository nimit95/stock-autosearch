# autoresearch

This is an experiment to have the LLM do its own research on ML algorithms for the Indian stock market.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data fetching/caching, stock universe, date ranges, evaluation (backtesting). Do not modify.
   - `train.py` — the file you modify. Feature engineering, target/label construction, model architecture, hyperparameters, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains cached market data. If not, tell the human to run `python prepare.py` to fetch and prepare Indian stock market data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs locally. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/data loading). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: feature engineering (add/remove/transform indicators), prediction targets/labels (1-day returns, 5-day returns, binary classification, rank targets), model type (XGBoost, LightGBM, neural nets, ensemble methods, etc.), hyperparameters, training strategy, position sizing logic, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation (backtesting), data fetching/caching, stock universe, date ranges, and training constants.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness. The `evaluate_strategy` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest sharpe_ratio on the test set.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the model, the features, the hyperparameters, the prediction target, the position sizing. The only constraint is that the code runs without crashing and finishes within the time budget.

**Overfitting** is the enemy. The evaluation is on an out-of-sample test period. A model that memorizes the training data but fails on the test set is worthless. Prefer robust, generalizable approaches.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 sharpe improvement that adds 50 lines of hacky code? Probably not worth it. A 0.01 sharpe improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
sharpe_ratio:     1.234567
alpha_pct:        12.5
total_return_pct: 45.2
benchmark_return_pct: 32.7
max_drawdown_pct: 12.3
win_rate_pct:     58.4
num_trades:       142
training_seconds: 300.1
total_seconds:    325.9
```

Alpha is the strategy's excess return over the benchmark (Nifty 50). `alpha_pct = total_return_pct - benchmark_return_pct`. A positive alpha means the model adds value beyond simply buying the index.

You can extract the key metrics from the log file:

```
grep "^sharpe_ratio:\|^alpha_pct:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	sharpe_ratio	total_return_pct	alpha_pct	max_drawdown_pct	status	description
```

1. git commit hash (short, 7 chars)
2. sharpe_ratio achieved (e.g. 1.234567) — use 0.000000 for crashes
3. total_return_pct (e.g. 45.2) — use 0.0 for crashes
4. alpha_pct over Nifty 50 (e.g. 12.5) — use 0.0 for crashes
5. max_drawdown_pct (e.g. 12.3) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	sharpe_ratio	total_return_pct	alpha_pct	max_drawdown_pct	status	description
a1b2c3d	0.850000	38.0	5.2	15.2	keep	baseline
b2c3d4e	1.120000	45.2	12.5	13.8	keep	switch to LightGBM with RSI features
c3d4e5f	0.720000	29.6	-3.1	22.1	discard	add sentiment features (overfit)
d4e5f6g	0.000000	0.0	0.0	0.0	crash	neural net too large (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar23`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^sharpe_ratio:\|^total_return_pct:\|^alpha_pct:\|^max_drawdown_pct:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If sharpe_ratio improved (higher), you "advance" the branch, keeping the git commit
9. If sharpe_ratio is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers on quantitative finance, re-read the in-scope files for new angles, try combining previous near-misses, try more radical model changes, explore different feature engineering approaches, different prediction horizons, ensemble methods. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
