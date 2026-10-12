"""One-command pipeline.

  python run_all.py --synthetic --fast     # smoke test, no downloads, ~1-3 min
  python run_all.py                        # real Kaggle data in data/raw/
  python run_all.py --tune                 # + Optuna/random-search tuning of RF & GB
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as C
from src import data_prep, explain, synthetic_data, train_recommend, train_yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic data first (testing only)")
    ap.add_argument("--fast", action="store_true", help="smaller models for quick runs")
    ap.add_argument("--tune", action="store_true", help="hyper-parameter tuning for RF / GB")
    ap.add_argument("--skip-explain", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        synthetic_data.make_all()
    data_prep.main()
    train_yield.main(fast=a.fast, do_tune=a.tune)
    train_recommend.main(fast=a.fast)
    if not a.skip_explain:
        explain.main()
    print("\nDone. Results -> reports/, models -> models/. Demo:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
