# step1_fred.py
# Downloads and cleans macroeconomic series from FRED for 2019-2024.
# Outputs: data/fred_daily.csv

import requests
import pandas as pd
from pathlib import Path
from io import StringIO

OUTPUT = Path("data")
OUTPUT.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"
END   = "2024-12-31"

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Series to download: {column_name: FRED_code}
SERIES = {
    "baa_treasury_spread": "BAA10Y",
    "aaa_treasury_spread": "AAA10Y",
    "moody_baa_yield":     "DBAA",
    "moody_aaa_yield":     "DAAA",
    "vix":                 "VIXCLS",
    "nfci":                "NFCI",
    "fed_funds":           "DFEDTARU",
    "t_3m":                "DGS3MO",
    "t_10y":               "DGS10",
    "term_spread":         "T10Y3M",
}


def fetch_series(code: str) -> pd.Series | None:
    """Download a single FRED series and return as a date-indexed Series."""
    resp = requests.get(f"{FRED_BASE}?id={code}", timeout=15)
    if resp.status_code != 200:
        print(f"  WARNING: {code} returned HTTP {resp.status_code}")
        return None
    df = pd.read_csv(StringIO(resp.text), na_values=".")
    df.columns = ["date", "value"]
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    mask = (df["date"] >= START) & (df["date"] <= END)
    return df[mask].set_index("date")["value"]


def main():
    print("Downloading FRED series...")
    frames = {}
    for name, code in SERIES.items():
        s = fetch_series(code)
        if s is not None and len(s) > 0:
            frames[name] = s
            print(f"  {name:30s} {len(s):,} obs   "
                  f"mean = {s.mean():.3f}")
        else:
            print(f"  {name:30s} FAILED")

    fred = pd.DataFrame(frames)
    fred.index.name = "date"

    # Derived spread
    if "moody_baa_yield" in fred.columns and "moody_aaa_yield" in fred.columns:
        fred["baa_aaa_spread"] = (fred["moody_baa_yield"]
                                  - fred["moody_aaa_yield"])

    # Forward-fill weekends / holidays
    fred = fred.ffill().reset_index()

    out_path = OUTPUT / "fred_daily.csv"
    fred.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  shape={fred.shape}")

    # Coverage summary
    print("\nCoverage:")
    for col in fred.columns:
        if col == "date":
            continue
        n   = fred[col].notna().sum()
        pct = n / len(fred) * 100
        mn  = fred[col].mean()
        print(f"  {col:30s} {n:,}/{len(fred):,} "
              f"({pct:.1f}%)  mean={mn:.3f}")

    print("\nDescriptive statistics:")
    print(fred.drop(columns="date").describe().round(4).to_string())


if __name__ == "__main__":
    main()
