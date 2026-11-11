Install
pip install -r requirements.txt

Tested on Python 3.12 (Linux) only. Not tested on 3.14 or Windows yet. If pip cannot install torch on your Python, use 3.12.

Run
python -m pytest -q          # checks the code works
python -m src.train --config config/run_c_final.yaml --seed 0 --quick   # short smoke test, NOT final results
python -m src.train --config config/run_c_final.yaml --seed 0           # full training run
tensorboard --logdir runs
Where the output goes
results/ CSV files
runs/ TensorBoard logs and trained models

Quick mode writes to results_quick/ and runs_quick/ instead, so it never mixes with the full run.

Rules the code follows
Everything is seeded. Seeds and library versions are saved with each run.
50 train, 20 validation and 20 test scenarios. The test set is used once; results/test_lock.json warns if it is used again.
The pass/fail limits are in config/run_c_final.yaml, written before any run. They are our own choice, not an official standard.