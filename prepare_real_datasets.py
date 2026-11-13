"""Turn a real sales / transactions file into ONE weekly demand series for demo.py.

Run from the project folder. Examples:

  # one row per sale, all products added together
  python prepare_real_datasets.py --input data/raw/sales.csv --date-col date --qty-col quantity

  # only one product (name its column and value)
  python prepare_real_datasets.py --input data/raw/sales.csv --date-col InvoiceDate \
      --qty-col Quantity --group-col StockCode --group-value 85123A

  # name a group column but no value: the busiest group is picked for you
  python prepare_real_datasets.py --input data/raw/sales.csv --date-col date \
      --qty-col sales --group-col item

  # UCI Online Retail II (online_retail_II.xlsx: two yearly sheets, both are read), all products added up
  python prepare_real_datasets.py --input data/raw/online_retail_II.xlsx --date-col InvoiceDate --qty-col Quantity

Output: data/processed/real_dataset.csv with two columns (week_ending, demand).
.csv and .xlsx are both accepted (.xlsx needs the openpyxl package).

What it does: drops returns (negative quantities), adds up the quantity per week
(weeks end on Sunday), fills weeks with no sales with 0, and drops the first and last week
because they are usually only partly covered by the file.
"""
import argparse
from pathlib import Path

import pandas as pd


def read_table(path):
    """Read a .csv, or every sheet of an .xlsx (the UCI file has one sheet per year)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        frames = []
        for i, (name, sheet) in enumerate(pd.read_excel(path, sheet_name=None).items()):
            print(f"Sheet '{name}': {len(sheet)} rows")
            frames.append(sheet.assign(_sheet=i))
        return pd.concat(frames, ignore_index=True)
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:                       # many retail files are not UTF-8
        df = pd.read_csv(path, encoding="latin-1")
    return df.assign(_sheet=0)


def drop_overlapping_sheets(df, date_col):
    """If a later sheet repeats days already covered by an earlier sheet, keep the earlier rows only."""
    if df["_sheet"].nunique() < 2:
        return df
    keep, latest = pd.Series(True, index=df.index), None
    for i in sorted(df["_sheet"].unique()):
        in_sheet = df["_sheet"] == i
        if latest is not None:
            keep &= ~(in_sheet & (df[date_col] <= latest))
        sheet_max = df.loc[in_sheet & keep, date_col].max()
        latest = sheet_max if latest is None else max(latest, sheet_max)
    if (~keep).any():
        print(f"Sheets overlap in time: removed {(~keep).sum()} repeated rows so no day is counted twice")
    return df[keep]


def check_columns(df, names):
    for name in names:
        if name and name not in df.columns:
            raise SystemExit(f"Column '{name}' not found. Columns in the file: {list(df.columns)}")


def pick_busiest_group(df, date_col, group_col):
    """The group (e.g. product) that sold in the most different weeks, ties broken by volume.
    Codes that do not start with a digit (POST, DOT, M, ...) are services, not products, so they are skipped."""
    weeks = df[date_col].dt.to_period("W")
    active_weeks = df.assign(_w=weeks).groupby(group_col)["_w"].nunique()
    volume = df.groupby(group_col)["_qty"].sum()
    table = pd.DataFrame({"weeks": active_weeks, "volume": volume})
    product_like = table.index.astype(str).str[:1].str.isdigit()
    if product_like.any():
        table = table[product_like]
    return table.sort_values(["weeks", "volume"], ascending=False).index[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="path to your sales file (.csv or .xlsx)")
    p.add_argument("--date-col", required=True, help="name of the date column")
    p.add_argument("--qty-col", required=True, help="name of the quantity / units sold column")
    p.add_argument("--group-col", help="optional: column to filter on, e.g. product or store")
    p.add_argument("--group-value", help="optional: the value to keep in --group-col")
    p.add_argument("--dayfirst", action="store_true", help="dates are written day/month/year")
    p.add_argument("--keep-edges", action="store_true", help="do not drop the first and last (partial) week")
    p.add_argument("--out", default="data/processed/real_dataset.csv")
    args = p.parse_args()

    df = read_table(args.input)
    check_columns(df, [args.date_col, args.qty_col, args.group_col])
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce", dayfirst=args.dayfirst)
    df["_qty"] = pd.to_numeric(df[args.qty_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[args.date_col, "_qty"])
    df = drop_overlapping_sheets(df, args.date_col)
    df = df[df["_qty"] > 0]                           # returns / corrections are not demand
    print(f"Rows read: {before}, rows kept (valid date, quantity > 0): {len(df)}")

    if args.group_col:
        value = args.group_value
        if value is None:
            value = pick_busiest_group(df, args.date_col, args.group_col)
            print(f"No --group-value given: using the busiest {args.group_col} = {value}")
        else:                                         # compare as text so numbers and codes both work
            df = df.assign(_g=df[args.group_col].astype(str))
            value = str(value)
            args.group_col = "_g"
        df = df[df[args.group_col].astype(str) == str(value)]
        if df.empty:
            raise SystemExit(f"No rows with {args.group_col} = {value}")

    weekly = df.set_index(args.date_col)["_qty"].resample("W-SUN").sum()
    if not args.keep_edges and len(weekly) > 2:
        weekly = weekly.iloc[1:-1]
    out = weekly.rename("demand").rename_axis("week_ending").reset_index()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    zero_share = (out["demand"] == 0).mean()
    print(f"\nSaved {len(out)} weeks to {args.out}")
    print(f"  from {out['week_ending'].iloc[0].date()} to {out['week_ending'].iloc[-1].date()}")
    print(f"  weekly demand: mean {out['demand'].mean():.1f}, min {out['demand'].min():.0f}, "
          f"max {out['demand'].max():.0f}, weeks with zero sales {zero_share:.0%}")
    peak = out["demand"].max() / max(out["demand"].mean(), 1e-9)
    print(f"  busiest week = {peak:.1f} x the average week (training scenarios peak at about 1.1 to 1.4 x)")
    if len(out) < 53:
        print("  WARNING: demo.py needs at least 53 weeks (52 + 1 for the forecast). Use a longer "
              "period, or add up more products (leave out --group-col).")
    else:
        print(f"  OK: {len(out) - 52} different 52-week windows are possible (demo.py --start 0 ... {len(out) - 53}).")
    if zero_share > 0.10:
        print("  NOTE: many weeks have zero sales (intermittent demand). The agent trained on smooth "
              "seasonal demand, so a busier product or a total over several products demos better.")


if __name__ == "__main__":
    main()