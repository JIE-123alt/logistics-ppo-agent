"""Prepare held-out product-level weekly demand for evaluation of synthetic-trained PPO."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_weekly_panel(path, date_col, qty_col, product_col):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    columns = [date_col, qty_col, product_col]
    if path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path, usecols=columns, dtype={product_col: str})
    else:
        frame = pd.read_csv(path, usecols=columns, dtype={product_col: str})

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[qty_col] = pd.to_numeric(frame[qty_col], errors="coerce")
    frame = frame.dropna(subset=columns)
    frame = frame[(frame[qty_col] > 0) & frame[product_col].str.match(r"^\d{5}")]
    if frame.empty:
        raise ValueError("No rows remain after keeping dated, positive-quantity product sales.")

    weekly = (frame.groupby([pd.Grouper(key=date_col, freq="W-SUN"), product_col])[qty_col]
              .sum().unstack(product_col).fillna(0.0))
    calendar = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-SUN")
    weekly = weekly.reindex(calendar, fill_value=0.0).iloc[1:-1]
    weekly.index.name = "week_ending"
    if len(weekly) < 60:
        raise ValueError(f"Only {len(weekly)} full weeks are available; need at least 60.")
    return weekly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Real sales CSV or XLSX")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--qty-col", required=True)
    parser.add_argument("--group-col", required=True, help="Product identifier column, e.g. StockCode")
    parser.add_argument("--out-dir", default="data/processed/real_panel")
    parser.add_argument("--calibration-weeks", type=int, default=12)
    parser.add_argument("--max-zero", type=float, default=0.20)
    parser.add_argument("--max-peak", type=float, default=4.0)
    parser.add_argument("--min-mean", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.calibration_weeks < 4:
        parser.error("--calibration-weeks must be at least 4 for the trailing forecast.")
    weekly = build_weekly_panel(args.input, args.date_col, args.qty_col, args.group_col)
    if args.calibration_weeks >= len(weekly) - 52:
        parser.error("--calibration-weeks leaves too little history for 52-week episodes.")

    calibration = weekly.iloc[:args.calibration_weeks]
    means = calibration.mean()
    profile = pd.DataFrame({
        "calibration_mean": means,
        "calibration_zero_share": (calibration == 0).mean(),
        "calibration_peak": calibration.max() / means.replace(0, np.nan),
    })
    eligible = profile[
        (profile["calibration_mean"] >= args.min_mean)
        & (profile["calibration_zero_share"] <= args.max_zero)
        & (profile["calibration_peak"] <= args.max_peak)
    ].dropna()
    if len(eligible) < 10:
        raise ValueError(
            f"Only {len(eligible)} products pass the calibration filters; "
            "relax --max-zero, --max-peak, or --min-mean."
        )

    products = np.random.default_rng(args.seed).permutation(eligible.index.astype(str).to_numpy())
    test_count = max(2, int(len(products) * 0.20))
    test_products = products[-test_count:]
    split = {"test": test_products.tolist()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = split["test"]
    long = (weekly[chosen].rename_axis("week_ending").reset_index()
            .melt(id_vars="week_ending", var_name="product", value_name="demand"))
    long.to_csv(out / "real_panel.csv", index=False)

    split_info = {
        "source": str(Path(args.input)),
        "weeks": len(weekly),
        "first_week": str(weekly.index[0].date()),
        "last_week": str(weekly.index[-1].date()),
        "calibration_weeks": args.calibration_weeks,
        "purpose": "held-out real-data evaluation only",
        "seed": args.seed,
        "filters": {
            "max_zero": args.max_zero,
            "max_peak": args.max_peak,
            "min_mean": args.min_mean,
        },
        "eligible_products": len(eligible),
        "test": split["test"],
    }
    (out / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    profile = eligible.loc[chosen].copy()
    profile["split"] = "test"
    profile.to_csv(out / "profile.csv", index_label="product")

    print(f"Weekly panel: {len(weekly)} weeks, {len(weekly.columns)} products")
    print(f"Eligible from first {args.calibration_weeks} calibration weeks: {len(eligible)}")
    print(f"Saved {len(chosen)} held-out test products to {out}")
    print("Only positive sales are treated as demand; negative returns/cancellations are excluded.")
    print("The held-out products share the same calendar years, so the test is about new products, not future years.")


if __name__ == "__main__":
    main()
