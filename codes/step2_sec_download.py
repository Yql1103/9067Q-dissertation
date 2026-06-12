# step2_sec_download.py
# Downloads 10-K, 10-Q, and 8-K filings from SEC EDGAR for all
# sample firms over 2019-2024.  Supports resume after interruption.
# Outputs: data/sec_filings/TICKER/FORM/DATE/
#          data/sec_filing_counts.csv

import json
import time
from pathlib import Path

import pandas as pd
from sec_edgar_downloader import Downloader

OUTPUT  = Path("data")
SEC_DIR = OUTPUT / "sec_filings"
SEC_DIR.mkdir(parents=True, exist_ok=True)

YOUR_NAME  = "Qilong Yu"
YOUR_EMAIL = "qy247@cam.ac.uk"

START = "2019-01-01"
END   = "2024-12-31"

FORMS = ["10-K", "10-Q", "8-K"]

TICKERS = [
    # Financials
    "JPM", "BAC", "GS", "MS", "AIG",
    # Information Technology
    "AAPL", "MSFT", "INTC", "IBM", "CSCO",
    # Industrials
    "GE", "BA", "CAT", "HON", "MMM",
    # Consumer Discretionary
    "F", "GM", "MCD", "NKE", "HD",
    # Health Care
    "JNJ", "PFE", "MRK", "ABT", "CVS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",
    # Communication Services
    "T", "VZ", "CMCSA", "NFLX",
    # Materials
    "DOW", "NEM", "APD",
    # Consumer Staples
    "KO", "WMT", "PG",
    # Utilities
    "NEE", "DUK",
    # Real Estate
    "PLD", "AMT",
]

PROGRESS_FILE = OUTPUT / "sec_download_progress.json"


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done: set) -> None:
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))


def count_filings() -> pd.DataFrame:
    """Return a DataFrame with filing counts by ticker and form."""
    rows = []
    for ticker in TICKERS:
        for form in FORMS:
            d = SEC_DIR / ticker / form
            n = sum(1 for x in d.iterdir() if x.is_dir()) if d.exists() else 0
            rows.append({"ticker": ticker, "form": form, "n_filings": n})
    return pd.DataFrame(rows)


def main():
    done  = load_progress()
    total = len(TICKERS) * len(FORMS)
    print(f"Total tasks : {total}")
    print(f"Already done: {len(done)}")
    print(f"Remaining   : {total - len(done)}\n")

    dl = Downloader(YOUR_NAME, YOUR_EMAIL, str(SEC_DIR))
    n  = 0

    for ticker in TICKERS:
        for form in FORMS:
            n   += 1
            key  = f"{ticker}_{form}"
            tag  = f"[{n:3d}/{total}]"

            if key in done:
                print(f"{tag} SKIP  {ticker:6s} {form}")
                continue

            try:
                print(f"{tag} GET   {ticker:6s} {form} ...",
                      end=" ", flush=True)
                dl.get(form, ticker,
                       after=START,
                       before=END,
                       download_details=True)
                done.add(key)
                print("ok")
            except Exception as exc:
                print(f"FAILED  {exc}")

            if n % 10 == 0:
                save_progress(done)

            time.sleep(0.12)   # respect SEC rate limit (10 req/s)

    save_progress(done)

    # Summary table
    counts = count_filings()
    pivot  = (counts
              .pivot_table(index="ticker",
                           columns="form",
                           values="n_filings")
              .fillna(0)
              .astype(int)
              .reset_index())
    pivot["Total"] = pivot[FORMS].sum(axis=1)
    pivot.to_csv(OUTPUT / "sec_filing_counts.csv", index=False)

    by_form = (counts
               .groupby("form")["n_filings"]
               .agg(total="sum", mean="mean",
                    min="min", max="max")
               .reset_index())

    print("\nFilings downloaded by form:")
    print(by_form.to_string(index=False))
    print(f"\nTotal filings: {pivot['Total'].sum()}")
    print(f"Saved: {OUTPUT}/sec_filing_counts.csv")


if __name__ == "__main__":
    main()
