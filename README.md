# FinBERT Analysis of SEC Filings: Disclosure Sentiment, Realized Volatility, and Credit Spread Transmission

**MPhil in Finance and Economics — University of Cambridge, July 2026**  
**Candidate:** Qilong Yu &nbsp;|&nbsp; **Supervisor:** Andrew Harvey

---

## Overview

This repository contains the data pipeline for the dissertation, which investigates whether FinBERT sentiment extracted from mandatory SEC filings contains predictive information for firm-level range-based volatility and aggregate credit spread dynamics.

The pipeline covers a panel of **41 U.S. firms** across **10 GICS sectors** over **2019–2024** (61,848 firm-day observations), and proceeds in four steps:

```
step1_fred.py        →  Macroeconomic controls (FRED)
step2_sec_download.py →  SEC 10-K / 8-K filings (EDGAR)
step3_finbert.py     →  FinBERT sentiment scoring
step4_wrds_panel.py  →  CRSP + Compustat + master panel
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
│   ├── sec_filings/
│   ├── finbert_scores.csv
│   ├── sentiment_firmday.csv
│   ├── crsp_daily.csv
│   ├── compustat_quarterly.csv
│   ├── crsp_compustat_link.csv
│   └── master_panel.csv
└── README.md
```

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
```

</details>

### WRDS access
Steps 1 and 4 require an active [WRDS](https://wrds-www.wharton.upenn.edu/) account with access to:
- `crsp.dsf` and `crsp.msenames` (CRSP daily stock file)
- `crsp.ccmxpf_linktable` (CRSP–Compustat link)
- `comp.fundq` and `comp.security` (Compustat quarterly)

On first run, `wrds.Connection()` will prompt for your WRDS credentials.

---

## Usage

Run the four steps in order. Each step caches its output to `data/` and supports **resume after interruption** — re-running a step will skip already-completed work.

### Step 1 — FRED macroeconomic data

Downloads Moody's BAA/AAA–Treasury spreads, VIX, NFCI, federal funds rate, and Treasury yields for 2019–2024.

```bash
python step1_fred.py
```

Output: `data/fred_daily.csv`

---

### Step 2 — SEC filing download

Downloads 10-K and 8-K filings for all 41 sample firms from SEC EDGAR. Progress is saved after every 10 tasks; the script can be safely interrupted and restarted.

```bash
python step2_sec_download.py
```

Output: `data/sec_filings/TICKER/FORM/DATE/`  
Summary: `data/sec_filing_counts.csv`

> **Note:** 10-Q filings are downloaded but excluded from sentiment scoring in Step 3, as fewer than 0.05% contain sufficient narrative text after HTML table removal.

---

### Step 3 — FinBERT sentiment scoring

Scores each filing using [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone). Documents exceeding the 512-token BERT limit are split into 510-token chunks; chunk-level probabilities are averaged. The document-level sentiment score is:

```
Sent = P(Positive) − P(Negative)  ∈ [−1, 1]
```

Label mapping (verified by unit tests): `0 = Neutral`, `1 = Positive`, `2 = Negative`.

```bash
python step3_finbert.py
```

Outputs:
| File | Description |
|---|---|
| `data/finbert_scores.csv` | Document-level scores |
| `data/sentiment_by_source.csv` | Summary by filing type (Table 3) |
| `data/sentiment_firmday.csv` | Firm-day aggregates with AvgSent, NegShare, TextVolume, SentMom, SentVol |
| `data/sentiment_stats.csv` | Descriptive statistics (Table 4) |

The script uses MPS (Apple Silicon), CUDA, or CPU automatically. A checkpoint file (`data/finbert_checkpoint.json`) enables resuming from the last saved position.

---

### Step 4 — WRDS data and master panel

Downloads CRSP daily equity data and Compustat quarterly fundamentals from WRDS, then merges all data sources into a single firm-day panel. Compustat variables are matched with a **45-day reporting lag** to avoid look-ahead bias.

```bash
python step4_wrds_panel.py
```

Outputs:
| File | Description |
|---|---|
| `data/crsp_daily.csv` | CRSP equity data with Parkinson RBV and HAR lags |
| `data/compustat_quarterly.csv` | Quarterly fundamentals and financial ratios |
| `data/crsp_compustat_link.csv` | CCM link table |
| `data/master_panel.csv` | Final merged panel (≈ 61,848 firm-day obs) |

---

## Key Variables

| Variable | Definition | Source |
|---|---|---|
| `rbv` | Parkinson range-based volatility: `(ln H − ln L)² / (4 ln 2)` | CRSP |
| `ln_rbv` | Log of `rbv` | Derived |
| `rbv_w` / `rbv_m` | 5-day / 22-day lagged mean of `rbv` (HAR components) | Derived |
| `avg_sent` | Mean FinBERT sentiment across same-day filings | EDGAR + FinBERT |
| `neg_share` | Fraction of filings with sentiment < −0.05 | Derived |
| `text_volume` | log(1 + number of filings) | Derived |
| `sent_mom` | 5-day sentiment momentum (winsorised 1/99%) | Derived |
| `sent_vol` | 5-day rolling std of `avg_sent` | Derived |
| `baa_treasury_spread` | Moody's BAA yield minus 10Y Treasury | FRED: BAA10Y |
| `leverage` | (Long-term debt + current debt) / total assets | Compustat |
| `profitability` | Operating income before depreciation / total assets | Compustat |

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

## Notes

- `data/` is excluded from version control (add to `.gitignore`). Raw filings can be large (several GB).
- FRED series are fetched without an API key via the public CSV endpoint; no registration is required.
- The SEC EDGAR downloader respects the 10 requests/second rate limit automatically.
- All scripts are idempotent: cached files are loaded from disk on subsequent runs.

---

\paragraph{SEC filings.}
Form 10-K and Form 8-K filings are downloaded from the
\textbf{SEC EDGAR} database (\url{https://www.sec.gov/cgi-bin/browse-edgar})
using the \texttt{sec-edgar-downloader} Python package. Filing dates
are extracted from the \texttt{FILED AS OF DATE} field in the SGML
submission header (\texttt{full-submission.txt}). Raw HTML documents
are cleaned with \texttt{BeautifulSoup} by removing financial
statement tables, XBRL-tagged data, navigation bars, and boilerplate
legal text. Documents are capped at 8,000 words and must contain at
least 150 clean words to be retained; Form 10-Q filings are excluded
because fewer than 0.05\% contain sufficient narrative text after
table removal. The baseline textual dataset comprises 263 Form~10-K
and 4,042 Form~8-K documents across 41 firms over 2019--2024.

\vspace{0.3em}
\noindent Due to the large size of the raw filing archive (several
gigabytes), the downloaded SEC filings are not included in the GitHub
repository. The complete filing archive is deposited separately and
is available for download at:

\begin{center}
\url{https://drive.google.com/file/d/1eRr2smRRcK8peeIzdGcPJwMEgd4kT-jO/view?usp=drive_link}
\end{center}

\noindent The archive should be extracted into the
\texttt{data/sec\_filings/} directory before running
\texttt{step3\_finbert.py}. The expected folder structure is:

\begin{verbatim}
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
\end{verbatim}

\noindent Alternatively, the filings can be re-downloaded from
scratch by running \texttt{step2\_sec\_download.py}, which fetches
all filings directly from SEC EDGAR and supports resuming after
interruption.
