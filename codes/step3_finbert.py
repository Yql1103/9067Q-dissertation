# step3_finbert.py
# Scores SEC 10-K and 8-K filings with FinBERT (yiyanghkust/finbert-tone).
# Supports resume after interruption via a checkpoint file.
# Outputs: data/finbert_scores.csv
#          data/sentiment_by_source.csv
#          data/sentiment_firmday.csv

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path("data")
SEC_DIR = BASE / "sec_filings" / "sec-edgar-filings"
OUTPUT  = BASE
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "yiyanghkust/finbert-tone"
# Label mapping verified by unit tests (Appendix A.2):
#   0 = Neutral, 1 = Positive, 2 = Negative
POS_IDX = 1
NEG_IDX = 2

MAX_CHUNKS  = 3       # chunks per document (3 × 510 tokens ≈ first 8k words)
MIN_WORDS   = 150     # minimum clean word count to retain a document
SAVE_EVERY  = 100     # checkpoint frequency (number of new documents)

FORMS = ["10-K", "8-K"]   # 10-Q excluded: <0.05% contain sufficient narrative

SECTOR_MAP = {
    # Financials
    "JPM": "Financials",  "BAC": "Financials",  "GS": "Financials",
    "MS":  "Financials",  "AIG": "Financials",
    # Information Technology
    "AAPL": "Information Tech.", "MSFT": "Information Tech.",
    "INTC": "Information Tech.", "IBM":  "Information Tech.",
    "CSCO": "Information Tech.",
    # Industrials
    "GE": "Industrials", "BA":  "Industrials", "CAT": "Industrials",
    "HON":"Industrials",  "MMM": "Industrials",
    # Consumer Discretionary
    "F":   "Consumer Discret.", "GM":  "Consumer Discret.",
    "MCD": "Consumer Discret.", "NKE": "Consumer Discret.",
    "HD":  "Consumer Discret.",
    # Health Care
    "JNJ": "Health Care", "PFE": "Health Care", "MRK": "Health Care",
    "ABT": "Health Care", "CVS": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "OXY": "Energy",
    # Communication Services
    "T": "Communication Svcs", "VZ":    "Communication Svcs",
    "CMCSA": "Communication Svcs",      "NFLX": "Communication Svcs",
    # Materials
    "DOW": "Materials", "NEM": "Materials", "APD": "Materials",
    # Consumer Staples
    "KO": "Consumer Staples", "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate",
}


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Text extraction ───────────────────────────────────────────────────────────
def _parse_html(path: Path) -> str | None:
    """Extract clean narrative text from an HTML filing document."""
    try:
        raw  = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "table",
                         "footer", "header", "nav"]):
            tag.decompose()
        text  = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            return None
        return " ".join(words[:8_000])
    except Exception:
        return None


def _parse_submission_txt(path: Path) -> str | None:
    """
    Extract narrative text from a full-submission.txt file.
    Strips SGML wrappers and embedded HTML tables before returning text.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # Extract content inside <TEXT> … </TEXT>
        match = re.search(r"<TEXT>(.*?)</TEXT>",
                          raw, re.DOTALL | re.IGNORECASE)
        if match:
            raw = match.group(1)
        # Parse embedded HTML if present
        if re.search(r"<html", raw, re.IGNORECASE):
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(["script", "style", "table",
                             "footer", "header"]):
                tag.decompose()
            raw = soup.get_text(" ")
        # Remove residual SGML tags
        text  = re.sub(r"<[^>]+>", " ", raw)
        text  = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            return None
        return " ".join(words[:8_000])
    except Exception:
        return None


def extract_text(filing_dir: Path) -> str | None:
    """
    Return narrative text for a filing directory.
    Priority: primary-document.html → full-submission.txt.
    """
    html_doc = filing_dir / "primary-document.html"
    if html_doc.exists():
        text = _parse_html(html_doc)
        if text and len(text.split()) >= 500:
            return text

    txt_doc = filing_dir / "full-submission.txt"
    if txt_doc.exists():
        return _parse_submission_txt(txt_doc)

    return None


def get_filing_date(filing_dir: Path) -> str:
    """
    Read the filing date from the SGML submission header.
    Returns ISO date string 'YYYY-MM-DD' or 'unknown'.
    """
    header = filing_dir / "full-submission.txt"
    if header.exists():
        try:
            with open(header, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i > 50:
                        break
                    if "FILED AS OF DATE:" in line:
                        d = line.split(":")[-1].strip()
                        if len(d) == 8:
                            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        except Exception:
            pass
    return "unknown"


# ── FinBERT scoring ───────────────────────────────────────────────────────────
def score_text(text: str,
               tokenizer: BertTokenizer,
               model: BertForSequenceClassification,
               device: torch.device) -> dict:
    """
    Score a document with FinBERT.
    The document is split into 510-token chunks; probabilities are averaged.
    Returns sentiment = P(Positive) - P(Negative).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i: i + 510]
              for i in range(0, len(tokens), 510)][:MAX_CHUNKS]
    probs  = []
    with torch.no_grad():
        for chunk in chunks:
            ids = torch.tensor(
                [[tokenizer.cls_token_id] + chunk
                 + [tokenizer.sep_token_id]]
            ).to(device)
            out = model(input_ids=ids,
                        attention_mask=torch.ones_like(ids))
            p   = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            probs.append(p)
    avg = np.mean(probs, axis=0)
    return {
        "p_neutral":  float(avg[0]),
        "p_positive": float(avg[POS_IDX]),
        "p_negative": float(avg[NEG_IDX]),
        "sentiment":  float(avg[POS_IDX] - avg[NEG_IDX]),
        "n_chunks":   len(chunks),
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────
def _winsorise(series: pd.Series,
               lower: float = 0.01,
               upper: float = 0.99) -> pd.Series:
    return series.clip(series.quantile(lower),
                       series.quantile(upper))


def aggregate_to_firm_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse document-level scores to firm × filing-date level and
    compute dynamic sentiment features (momentum, volatility).
    """
    agg = (df.groupby(["ticker", "sector", "filing_date"])["sentiment"]
             .agg(avg_sent="mean",
                  neg_share=lambda x: (x < -0.05).mean(),
                  n_docs="count")
             .reset_index()
             .rename(columns={"filing_date": "date"}))

    agg["text_volume"] = np.log1p(agg["n_docs"])
    agg = agg.sort_values(["ticker", "date"])

    # 5-day rolling mean of past sentiment (excluding current day)
    roll_mean = (agg.groupby("ticker")["avg_sent"]
                    .transform(lambda x:
                               x.shift(1).rolling(5, min_periods=2).mean()))

    agg["sent_mom"] = ((agg["avg_sent"] - roll_mean)
                       / roll_mean.abs().clip(lower=0.01))
    agg["sent_mom"] = _winsorise(agg["sent_mom"])

    agg["sent_vol"] = (agg.groupby("ticker")["avg_sent"]
                          .transform(lambda x:
                                     x.rolling(5, min_periods=3).std()))
    return agg


def descriptive_stats(series: pd.Series, label: str) -> dict:
    s = series.dropna()
    return {
        "Variable": label,
        "N":        f"{len(s):,}",
        "Mean":     round(s.mean(), 4),
        "Std":      round(s.std(), 4),
        "Min":      round(s.min(), 4),
        "P25":      round(s.quantile(0.25), 4),
        "Median":   round(s.median(), 4),
        "Max":      round(s.max(), 4),
        "Skew":     round(s.skew(), 3),
        "Kurt":     round(s.kurtosis(), 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading FinBERT...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model     = model.to(device)
    model.eval()
    print("FinBERT ready\n")

    # Resume support
    scores_file = OUTPUT / "finbert_scores.csv"
    ckpt_file   = OUTPUT / "finbert_checkpoint.json"

    scored_hashes: set = (set(json.loads(ckpt_file.read_text()))
                          if ckpt_file.exists() else set())
    results: list = (pd.read_csv(scores_file).to_dict("records")
                     if scores_file.exists() else [])
    n_new = 0

    tickers = sorted([
        d.name for d in SEC_DIR.iterdir()
        if d.is_dir() and d.name in SECTOR_MAP
    ])
    print(f"Firms to process : {len(tickers)}")
    print(f"Already scored   : {len(scored_hashes)} documents\n")

    for ticker in tqdm(tickers, desc="Firms"):
        ticker_dir = SEC_DIR / ticker
        for form in FORMS:
            form_dir = ticker_dir / form
            if not form_dir.exists():
                continue

            for filing_dir in sorted(form_dir.iterdir()):
                if not filing_dir.is_dir():
                    continue

                doc_hash = hashlib.md5(
                    str(filing_dir).encode()).hexdigest()
                if doc_hash in scored_hashes:
                    continue

                filing_date = get_filing_date(filing_dir)
                text        = extract_text(filing_dir)

                if text is None:
                    scored_hashes.add(doc_hash)
                    continue

                try:
                    scores = score_text(text, tokenizer, model, device)
                    scores.update({
                        "ticker":      ticker,
                        "sector":      SECTOR_MAP[ticker],
                        "form":        form,
                        "filing_date": filing_date,
                        "n_words":     len(text.split()),
                        "doc_hash":    doc_hash,
                    })
                    results.append(scores)
                    n_new += 1
                except Exception as exc:
                    tqdm.write(f"  Error {ticker} {form}: {exc}")
                finally:
                    scored_hashes.add(doc_hash)

                if n_new % SAVE_EVERY == 0 and n_new > 0:
                    pd.DataFrame(results).to_csv(scores_file, index=False)
                    ckpt_file.write_text(json.dumps(sorted(scored_hashes)))
                    tqdm.write(f"  Checkpoint saved ({n_new} new)")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(scores_file, index=False)
    ckpt_file.write_text(json.dumps(sorted(scored_hashes)))
    print(f"\nTotal documents: {len(df):,}  |  New this run: {n_new}")

    if len(df) == 0:
        print("WARNING: no documents scored — check SEC_DIR structure.")
        return

    # Date filter
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.dropna(subset=["filing_date"])
    df = df[(df["filing_date"] >= "2019-01-01") &
            (df["filing_date"] <= "2024-12-31")]
    print(f"After date filter: {len(df):,}")

    # Validation
    print(f"\nSentiment mean : {df['sentiment'].mean():.4f}")
    print(f"Negative share : {(df['sentiment'] < 0).mean():.3f}")

    # Table A: sentiment by filing type
    table_a = (df.groupby("form")["sentiment"]
                 .agg(n_docs="count",
                      mean="mean",
                      std="std",
                      min="min",
                      p25=lambda x: x.quantile(0.25),
                      median="median",
                      max="max",
                      neg_share=lambda x: (x < -0.05).mean())
                 .reset_index()
                 .round(4))
    table_a.to_csv(OUTPUT / "sentiment_by_source.csv", index=False)
    print("\nSentiment by filing type:")
    print(table_a.to_string(index=False))

    # Aggregate to firm-day
    agg = aggregate_to_firm_day(df)
    agg.to_csv(OUTPUT / "sentiment_firmday.csv", index=False)

    # Table B: firm-day descriptive statistics
    table_b = pd.DataFrame([
        descriptive_stats(agg["avg_sent"],    "AvgSent (firm-day)"),
        descriptive_stats(agg["neg_share"],   "NegShare"),
        descriptive_stats(agg["text_volume"], "TextVolume log(1+N)"),
        descriptive_stats(agg["sent_mom"],    "SentMomentum (wins.)"),
        descriptive_stats(agg["sent_vol"],    "SentVol (5-day std)"),
    ])
    table_b.to_csv(OUTPUT / "sentiment_stats.csv", index=False)
    print("\nFirm-day sentiment statistics:")
    print(table_b.to_string(index=False))

    print("\nOutputs saved to data/:")
    print("  finbert_scores.csv")
    print("  sentiment_by_source.csv")
    print("  sentiment_firmday.csv")
    print("  sentiment_stats.csv")


if __name__ == "__main__":
    main()
