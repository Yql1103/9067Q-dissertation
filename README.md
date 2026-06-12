# FinBERT Analysis of SEC Filings: Disclosure Sentiment, Realized Volatility, and Credit Spread Transmission

**MPhil in Finance and Economics — University of Cambridge, July 2026**  

---

## Overview

This repository contains the data pipeline for the dissertation, which investigates whether FinBERT sentiment extracted from mandatory SEC filings contains predictive information for firm-level range-based volatility and aggregate credit spread dynamics.

The pipeline covers a panel of **41 U.S. firms** across **10 GICS sectors** over **2019–2024** (61,848 firm-day observations), and proceeds in four steps:

```
step1_fred.py         →  Macroeconomic controls (FRED)
step2_sec_download.py →  SEC 10-K / 8-K filings (EDGAR)
step3_finbert.py      →  FinBERT sentiment scoring
step4_wrds_panel.py   →  CRSP + Compustat + master panel
```

---

## Repository Structure

```
.
├── step1_fred.py           # Download FRED macro series
├── step2_sec_download.py   # Download SEC filings via EDGAR
├── step3_finbert.py        # Score filings with FinBERT
├── step4_wrds_panel.py     # WRDS data + master panel assembly
├── data/                   # Generated outputs (not tracked by git)
│   ├── fred_daily.csv
│   ├── sec_filings/        # See "SEC Filing Archive" below
│   ├── finbert_scores.csv
│   ├── sentiment_firmday.csv
│   ├── crsp_daily.csv
│   ├── compustat_quarterly.csv
│   ├── crsp_compustat_link.csv
│   └── master_panel.csv
└── README.md
```

---

## SEC Filing Archive

Due to the large size of the raw filing archive (~several GB), the downloaded SEC filings are **not included in this repository**. The complete archive is available for download from Google Drive:

> **[Download SEC filings archive (Google Drive)](https://drive.google.com/file/d/1eRr2smRRcK8peeIzdGcPJwMEgd4kT-jO/view?usp=drive_link)**

The archive contains 263 Form 10-K and 4,042 Form 8-K filings across 41 firms over 2019–2024. After downloading, extract it so that the folder structure matches:

```
data/
└── sec_filings/
    └── sec-edgar-filings/
        └── TICKER/
            ├── 10-K/
            │   └── YYYY-MM-DD/
            │       ├── primary-document.html
            │       └── full-submission.txt
            └── 8-K/
                └── YYYY-MM-DD/
                    ├── primary-document.html
                    └── full-submission.txt
```

Once extracted, skip Step 2 and proceed directly to Step 3.

Alternatively, the filings can be re-downloaded from scratch by running `step2_sec_download.py`, which fetches all filings directly from SEC EDGAR and supports resuming after interruption.

---

## Requirements

### Python version
Python 3.10 or later.

### Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
numpy
pandas
torch
transformers
beautifulsoup4
lxml
tqdm
requests
wrds
sec-edgar-downloader
statsmodels
```

</details>

### WRDS access
Step 4 requires an active [WRDS](https://wrds-www.wharton.upenn.edu/) account with access to:
- `crsp.dsf` and `crsp.msenames` — CRSP daily stock file
- `crsp.ccmxpf_linktable` — CRSP–Compustat CCM link table
- `comp.fundq` and `comp.security` — Compustat quarterly fundamentals

On first run, `wrds.Connection()` will prompt for your WRDS credentials and cache them locally.

> **Note:** Steps 1–3 do not require WRDS access. Step 1 uses the public FRED CSV endpoint (no API key needed). Step 2 uses the SEC EDGAR public API.

---

## Usage

Run the four steps in order. Each step caches its output to `data/` and is **idempotent** — re-running will load cached files from disk rather than re-downloading.

### Step 1 — FRED macroeconomic data

Downloads Moody's BAA/AAA–Treasury spreads, VIX, NFCI, federal funds rate, and Treasury yields for 2019–2024.

```bash
python step1_fred.py
```

| Output | Description |
|---|---|
| `data/fred_daily.csv` | Daily macro series, forward-filled for weekends/holidays |

---

### Step 2 — SEC filing download

Downloads 10-K and 8-K filings for all 41 sample firms from SEC EDGAR. Progress is saved after every 10 tasks; the script can be safely interrupted and restarted.

```bash
python step2_sec_download.py
```

| Output | Description |
|---|---|
| `data/sec_filings/TICKER/FORM/DATE/` | Raw HTML filing documents |
| `data/sec_filing_counts.csv` | Filing count by firm and form type |

> **Skip this step** if you have downloaded the filing archive from Google Drive (see [SEC Filing Archive](#sec-filing-archive) above).

> **Note:** 10-Q filings are downloaded but excluded from sentiment scoring in Step 3, as fewer than 0.05% contain sufficient narrative text after HTML table removal.

---

### Step 3 — FinBERT sentiment scoring

Scores each filing using [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone). Documents exceeding the 512-token BERT input limit are split into non-overlapping 510-token chunks; chunk-level softmax probabilities are averaged to produce the document-level score:

```
Sent = P(Positive) − P(Negative)  ∈ [−1, 1]
```

Label mapping verified by unit tests on known-polarity financial sentences:

| Label index | Class | Test sentence |
|---|---|---|
| 0 | Neutral | "The board held its quarterly meeting." |
| 1 | Positive | "Earnings exceeded expectations, strong growth outlook." |
| 2 | Negative | "The company faces severe liquidity risk and may default." |

```bash
python step3_finbert.py
```

| Output | Description |
|---|---|
| `data/finbert_scores.csv` | Document-level FinBERT scores |
| `data/sentiment_by_source.csv` | Summary statistics by filing type |
| `data/sentiment_firmday.csv` | Firm-day aggregates (AvgSent, NegShare, TextVolume, SentMom, SentVol) |
| `data/sentiment_stats.csv` | Descriptive statistics |
| `data/finbert_checkpoint.json` | Resume checkpoint |

The script automatically selects the best available device (MPS → CUDA → CPU). A checkpoint file enables resuming from the last saved position after any interruption.

---

### Step 4 — WRDS data and master panel

Downloads CRSP daily equity data and Compustat quarterly fundamentals from WRDS, then merges all data sources (CRSP + FRED + Compustat + FinBERT sentiment) into a single firm-day panel. Compustat variables are matched with a **45-day reporting lag** to avoid look-ahead bias.

```bash
python step4_wrds_panel.py
```

| Output | Description |
|---|---|
| `data/crsp_daily.csv` | CRSP equity data with Parkinson RBV and HAR lag components |
| `data/compustat_quarterly.csv` | Quarterly fundamentals and financial ratios |
| `data/crsp_compustat_link.csv` | CCM link table (PERMNO ↔ GVKEY) |
| `data/master_panel.csv` | Final merged panel (≈ 61,848 firm-day obs × all variables) |

---

## Key Variables

### Volatility

| Variable | Definition | Source |
|---|---|---|
| `rbv` | Parkinson range-based volatility: `(ln H − ln L)² / (4 ln 2)` | CRSP |
| `ln_rbv` | Natural log of `rbv` | Derived |
| `rbv_w` | 5-day lagged mean of `rbv` (weekly HAR component) | Derived |
| `rbv_m` | 22-day lagged mean of `rbv` (monthly HAR component) | Derived |

### Sentiment

| Variable | Definition | Source |
|---|---|---|
| `avg_sent` | Mean `Sent` across same-day filings for firm `i` | EDGAR + FinBERT |
| `neg_share` | Fraction of filings with `Sent < −0.05` | Derived |
| `text_volume` | `log(1 + number of filings)` | Derived |
| `sent_mom` | 5-day sentiment momentum, winsorised at 1st/99th percentile | Derived |
| `sent_vol` | 5-day rolling standard deviation of `avg_sent` | Derived |

### Credit and macro

| Variable | FRED code | Description |
|---|---|---|
| `baa_treasury_spread` | `BAA10Y` | Moody's BAA yield minus 10Y Treasury (%) |
| `aaa_treasury_spread` | `AAA10Y` | Moody's AAA yield minus 10Y Treasury (%) |
| `vix` | `VIXCLS` | CBOE Volatility Index |
| `nfci` | `NFCI` | Chicago Fed National Financial Conditions Index |
| `fed_funds` | `DFEDTARU` | Federal funds upper target rate (%) |
| `t_3m` | `DGS3MO` | 3-month Treasury yield (%) |
| `t_10y` | `DGS10` | 10-year Treasury yield (%) |
| `term_spread` | `T10Y3M` | 10Y minus 3M Treasury yield (%) |

### Firm fundamentals (Compustat)

| Variable | Definition |
|---|---|
| `leverage` | (Long-term debt + current debt) / total assets |
| `profitability` | Operating income before depreciation / total assets |
| `current_ratio` | Current assets / current liabilities |
| `roe` | Net income / common equity |
| `log_assets` | Natural log of total assets |
| `rev_growth` | Quarter-on-quarter revenue growth |

All Compustat variables are winsorised at the 1st and 99th percentiles.

---

## Sample Firms

41 firms across 10 GICS sectors, all with market capitalisation ≥ $500 million throughout 2019–2024.

| Sector | Tickers |
|---|---|
| Financials | JPM, BAC, GS, MS, AIG |
| Information Technology | AAPL, MSFT, INTC, IBM, CSCO |
| Industrials | GE, BA, CAT, HON, MMM |
| Consumer Discretionary | F, GM, MCD, NKE, HD |
| Health Care | JNJ, PFE, MRK, ABT, CVS |
| Energy | XOM, CVX, COP, SLB, OXY |
| Communication Services | T, VZ, CMCSA, NFLX |
| Materials | DOW, NEM, APD |
| Consumer Staples | KO, WMT, PG |
| Utilities | NEE, DUK |

---

## Reproducing the Results

A complete replication proceeds as follows:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FRED macro data (no credentials required)
python step1_fred.py

# 3a. Download SEC filing archive from Google Drive and extract to data/sec_filings/
#     OR re-download from EDGAR (takes ~30–60 minutes):
python step2_sec_download.py

# 4. Score filings with FinBERT (GPU recommended; supports resume)
python step3_finbert.py

# 5. Download WRDS data and build master panel (WRDS credentials required)
python step4_wrds_panel.py
```

The master panel at `data/master_panel.csv` is the input to all econometric models in the dissertation (HAR, DCS-GB2, credit spread regression, and VAR).

---

## .gitignore

Add the following to `.gitignore` to avoid committing large data files:

```
data/
*.csv
*.json
__pycache__/
*.pyc
```

---


