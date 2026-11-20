**Install**
pip install -r requirements.txt

If pip cannot install torch on your Python, use 3.12.

**Run**
python -m pytest -q          # checks the code works
python -m src.train --run run_c_final --seed 0 --quick   # short smoke test, NOT final results
python -m src.train --run config/run_c_final.yaml --seed 0           # full training run
tensorboard --logdir runs
**Where the output goes**
results/ CSV files
runs/ TensorBoard logs and trained models

Quick mode writes to results_quick/ and runs_quick/ instead, so it never mixes with the full run.

## Held-out real-demand evaluation (no real-data training)

The PPO agent is trained only on synthetic demand using the regular `src.train` workflow. Real retail data
is kept separate and used only to evaluate frozen synthetic-trained Run C checkpoints.

1. Prepare a product-level weekly panel and a held-out test-product split:

   `python prepare_real_panel.py --input data/raw/online_retail_II.csv --date-col InvoiceDate --qty-col Quantity --group-col StockCode`

2. Evaluate the synthetic-trained checkpoints and synthetic-tuned baselines on the real test products:

   `python evaluate_real.py`

The evaluator reads only the `test` products from `data/processed/real_panel/split.json`; it does not
train or tune any policy on real data. It writes per-episode and summary metrics under `results_real/`.
Previous real-training reports were removed, but the real test products were evaluated in an earlier run.
The historical use is retained in `results_real/test_lock_real.json`; do not present this as a fresh,
untouched test. Further evaluations require `--allow-test-reuse`.

Demand is scaled using an initial calibration period to match the simulator's scale, so reported units are
simulator units rather than original retail quantities. Forecasts use earlier weeks only. The data contains
sales, not observed unmet demand, inventory, or supplier records; supplier behavior, lead times, costs, and
failures remain simulated. This tests generalization to held-out products from the same calendar period,
not future years. Multiple windows from a product overlap and are not independent observations. Disclose
AI assistance according to the course policy.

**Rules the code follows
**Everything is seeded. Seeds and library versions are saved with each run.
50 train, 20 validation and 20 test scenarios. The test set is used once; results/test_lock.json warns if it is used again.
The pass/fail limits are in config/run_c_final.yaml, written before any run. They are our own choice, not an official standard.
