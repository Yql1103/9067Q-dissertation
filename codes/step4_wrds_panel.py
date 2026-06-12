# step4_wrds_panel.py
# Downloads CRSP equity data and Compustat quarterly fundamentals from WRDS,
# then merges them with FRED macroeconomic variables and FinBERT sentiment
# to produce a master firm-day panel.
# Outputs: data/crsp_daily.csv
#          data/compustat_quarterly.csv
#          data/crsp_compustat_link.csv
#          data/master_panel.csv

import numpy as np
import pandas as pd
import wrds
from pathlib import Path

OUTPUT = Path("data")
OUTPUT.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"
END   = "2024-12-31"

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

TICKER_SQL = ",".join(f"'{t}'" for t in TICKERS)


# ── CRSP ──────────────────────────────────────────────────────────────────────
def download_crsp(conn: wrds.Connection) -> pd.DataFrame:
    """
    Download daily CRSP data and compute Parkinson range-based volatility (RBV)
    and HAR lag components.
    """
    query = f"""
        SELECT a.permno, a.date,
               a.ret, a.vol, a.prc,
               a.shrout, a.askhi, a.bidlo,
               b.ticker, b.comnam, b.hsiccd
        FROM crsp.dsf AS a
        LEFT JOIN crsp.msenames AS b
            ON  a.permno  = b.permno
            AND b.namedt <= a.date
            AND a.date   <= b.nameendt
        WHERE a.date BETWEEN '{START}' AND '{END}'
          AND b.shrcd   IN (10, 11)
          AND b.exchcd  IN (1, 2, 3)
          AND b.ticker  IN ({TICKER_SQL})
          AND a.prc     IS NOT NULL
          AND ABS(a.prc) > 0
    """
    print("  Downloading CRSP daily data...")
    crsp = conn.raw_sql(query, date_cols=["date"])
    print(f"  Raw: {crsp.shape[0]:,} obs, "
          f"{crsp['ticker'].nunique()} tickers")

    # Parkinson (1980) range-based volatility
    crsp = crsp.sort_values(["ticker", "date"])
    crsp["askhi"] = pd.to_numeric(crsp["askhi"], errors="coerce")
    crsp["bidlo"] = pd.to_numeric(crsp["bidlo"], errors="coerce")
    crsp["prc"]   = crsp["prc"].abs()

    valid = ((crsp["askhi"] > 0) &
             (crsp["bidlo"] > 0) &
             (crsp["askhi"] >= crsp["bidlo"]))
    crsp.loc[valid, "rbv"] = (
        (np.log(crsp.loc[valid, "askhi"])
         - np.log(crsp.loc[valid, "bidlo"])) ** 2
        / (4 * np.log(2))
    )

    # Market-cap filter (>= $500 million)
    crsp["mktcap"] = crsp["prc"].abs() * crsp["shrout"] / 1_000
    crsp = crsp[crsp["mktcap"] >= 500].copy()

    # Log return and log RBV
    crsp["logret"] = (crsp.groupby("ticker")["prc"]
                         .transform(lambda x: np.log(x).diff()))
    crsp["ln_rbv"] = np.log(crsp["rbv"].replace(0, np.nan))

    # HAR lag components (shift(1) ensures no same-day lookahead)
    def _roll_mean(x: pd.Series, window: int, min_p: int) -> pd.Series:
        return x.shift(1).rolling(window, min_periods=min_p).mean()

    crsp["rbv_lag1"] = crsp.groupby("ticker")["rbv"].shift(1)
    crsp["rbv_w"]    = crsp.groupby("ticker")["rbv"].transform(
        lambda x: _roll_mean(x, 5, 3))
    crsp["rbv_m"]    = crsp.groupby("ticker")["rbv"].transform(
        lambda x: _roll_mean(x, 22, 10))

    return crsp


# ── Compustat ─────────────────────────────────────────────────────────────────
def download_compustat(conn: wrds.Connection) -> pd.DataFrame:
    """
    Download quarterly Compustat fundamentals and construct financial ratios.
    A 45-day reporting lag is applied to avoid look-ahead bias.
    """
    query = f"""
        SELECT f.gvkey, f.datadate,
               f.atq, f.ltq, f.dlttq, f.dlcq,
               f.oibdpq, f.niq, f.ceqq,
               f.actq, f.lctq, f.revtq,
               s.tic AS ticker, s.cusip
        FROM comp.fundq AS f
        LEFT JOIN comp.security AS s
            ON  f.gvkey   = s.gvkey
            AND s.excntry = 'USA'
        WHERE f.datadate BETWEEN '2018-10-01' AND '{END}'
          AND f.indfmt  = 'INDL'
          AND f.datafmt = 'STD'
          AND f.popsrc  = 'D'
          AND f.consol  = 'C'
          AND f.atq     > 0
          AND s.tic     IN ({TICKER_SQL})
    """
    print("  Downloading Compustat quarterly data...")
    comp = conn.raw_sql(query, date_cols=["datadate"])
    print(f"  Raw: {comp.shape[0]:,} obs, "
          f"{comp['ticker'].nunique()} tickers")

    # Financial ratios
    comp["leverage"]     = ((comp["dlttq"].fillna(0)
                             + comp["dlcq"].fillna(0))
                            / comp["atq"])
    comp["profitability"] = comp["oibdpq"].fillna(0) / comp["atq"]
    comp["current_ratio"] = (comp["actq"].fillna(0)
                             / comp["lctq"].replace(0, np.nan))
    comp["roe"]           = (comp["niq"].fillna(0)
                             / comp["ceqq"].replace(0, np.nan))
    comp["log_assets"]    = np.log(comp["atq"])
    comp["rev_growth"]    = (comp.groupby("gvkey")["revtq"]
                                 .pct_change())

    # Winsorise at 1st/99th percentiles
    for col in ["leverage", "profitability", "current_ratio", "roe"]:
        lo, hi = comp[col].quantile(0.01), comp[col].quantile(0.99)
        comp[col] = comp[col].clip(lo, hi)

    # 45-day reporting lag: earliest date on which data are available
    comp["match_date"] = comp["datadate"] + pd.Timedelta(days=45)

    return comp


# ── CRSP–Compustat link ───────────────────────────────────────────────────────
def download_link(conn: wrds.Connection) -> pd.DataFrame:
    """Download the CRSP–Compustat CCM link table."""
    query = """
        SELECT gvkey, lpermno AS permno,
               linktype, linkprim,
               linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """
    print("  Downloading CCM link table...")
    link = conn.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    return link


# ── Panel assembly ────────────────────────────────────────────────────────────
def build_panel(crsp: pd.DataFrame,
                comp: pd.DataFrame,
                link: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CRSP, FRED, Compustat, and FinBERT sentiment into a master panel.
    Compustat is merged as-of with a 45-day reporting lag.
    """
    # Load FRED and sentiment
    fred_cols = [
        "date", "vix", "baa_treasury_spread", "aaa_treasury_spread",
        "baa_aaa_spread", "fed_funds", "t_3m", "t_10y",
        "term_spread", "nfci",
    ]
    fred = pd.read_csv(OUTPUT / "fred_daily.csv",
                       parse_dates=["date"],
                       usecols=lambda c: c in fred_cols)
    sent = pd.read_csv(OUTPUT / "sentiment_firmday.csv",
                       parse_dates=["date"])

    print(f"\n  CRSP rows      : {crsp.shape[0]:,}")
    print(f"  FRED rows      : {fred.shape[0]:,}")
    print(f"  Sentiment rows : {sent.shape[0]:,}")

    # Step 1: CRSP + FRED
    panel = crsp.merge(fred, on="date", how="left")

    # Step 2: attach gvkey via time-bounded CCM link
    crsp_link = (
        crsp[["permno", "date"]].drop_duplicates()
        .merge(link[["permno", "gvkey", "linkdt", "linkenddt"]],
               on="permno", how="left")
    )
    crsp_link = crsp_link[
        (crsp_link["date"] >= crsp_link["linkdt"]) &
        (crsp_link["date"] <= crsp_link["linkenddt"])
    ][["permno", "date", "gvkey"]].drop_duplicates()

    panel = panel.merge(crsp_link, on=["permno", "date"], how="left")
    pct   = panel["gvkey"].notna().mean() * 100
    print(f"  gvkey matched  : {pct:.1f}%")

    # Step 3: Compustat as-of merge (backward direction, 45-day lag)
    comp_clean = (
        comp[["gvkey", "match_date", "leverage", "profitability",
              "current_ratio", "log_assets", "roe", "rev_growth"]]
        .dropna(subset=["gvkey", "match_date"])
        .sort_values("match_date")
        .rename(columns={"match_date": "date"})
    )
    panel = pd.merge_asof(
        panel.sort_values("date"),
        comp_clean,
        on="date",
        by="gvkey",
        direction="backward",
    )
    pct = panel["leverage"].notna().mean() * 100
    print(f"  Compustat cov. : {pct:.1f}%")

    # Step 4: FinBERT sentiment
    sent_cols = [c for c in
                 ["ticker", "date", "avg_sent", "neg_share",
                  "text_volume", "sent_mom", "sent_vol"]
                 if c in sent.columns]
    panel = panel.merge(sent[sent_cols],
                        on=["ticker", "date"], how="left")
    pct = panel["avg_sent"].notna().mean() * 100
    print(f"  Sentiment cov. : {pct:.1f}%")

    return panel


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Connecting to WRDS...")
    conn = wrds.Connection()
    print("Connected\n")

    # CRSP
    crsp_path = OUTPUT / "crsp_daily.csv"
    if crsp_path.exists():
        print("CRSP daily: loading from cache...")
        crsp = pd.read_csv(crsp_path, parse_dates=["date"])
    else:
        crsp = download_crsp(conn)
        crsp.to_csv(crsp_path, index=False)
        print(f"  Saved: crsp_daily.csv  {crsp.shape}")

    print(f"  RBV mean: {crsp['rbv'].dropna().mean():.6f}")
    print(f"  RBV skew: {crsp['rbv'].dropna().skew():.3f}")
    print(f"  RBV kurt: {crsp['rbv'].dropna().kurtosis():.3f}")

    # Compustat
    comp_path = OUTPUT / "compustat_quarterly.csv"
    if comp_path.exists():
        print("\nCompustat quarterly: loading from cache...")
        comp = pd.read_csv(comp_path,
                           parse_dates=["datadate", "match_date"])
    else:
        comp = download_compustat(conn)
        comp.to_csv(comp_path, index=False)
        print(f"  Saved: compustat_quarterly.csv  {comp.shape}")

    print(f"  Leverage mean     : {comp['leverage'].mean():.3f}")
    print(f"  Profitability mean: {comp['profitability'].mean():.3f}")

    # CCM link
    link_path = OUTPUT / "crsp_compustat_link.csv"
    if link_path.exists():
        print("\nCCM link table: loading from cache...")
        link = pd.read_csv(link_path,
                           parse_dates=["linkdt", "linkenddt"])
    else:
        link = download_link(conn)
        link.to_csv(link_path, index=False)
        print(f"  Saved: crsp_compustat_link.csv  {link.shape}")

    conn.close()
    print("\nWRDS connection closed.")

    # Build master panel
    print("\nBuilding master panel...")
    panel = build_panel(crsp, comp, link)

    out_path = OUTPUT / "master_panel.csv"
    panel.to_csv(out_path, index=False)

    # Summary
    print("\n" + "=" * 55)
    print("MASTER PANEL SUMMARY")
    print("=" * 55)
    print(f"Shape    : {panel.shape[0]:,} obs × {panel.shape[1]} cols")
    print(f"Tickers  : {panel['ticker'].nunique()}")
    print(f"Date     : {panel['date'].min().date()} — "
          f"{panel['date'].max().date()}")

    print("\nCoverage:")
    for col in ["rbv", "ln_rbv", "logret", "leverage",
                "profitability", "avg_sent",
                "baa_treasury_spread", "vix"]:
        if col in panel.columns:
            pct = panel[col].notna().mean() * 100
            print(f"  {col:25s} {pct:5.1f}%")

    print("\nKey statistics:")
    for col, label in [
        ("rbv",           "RBV (Parkinson) mean"),
        ("ln_rbv",        "ln(RBV) mean"),
        ("leverage",      "Leverage mean"),
        ("profitability", "Profitability mean"),
        ("avg_sent",      "AvgSent mean"),
    ]:
        if col in panel.columns:
            s   = panel[col].dropna()
            print(f"  {label:30s} "
                  f"mean={s.mean():.4f}  "
                  f"skew={s.skew():.3f}  "
                  f"kurt={s.kurtosis():.3f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
