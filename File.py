import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Config
# ----------------------------

@dataclass
class RunConfig:
    ticker: str = "AAPL"
    min_days: int = 7               # ignore expirations too close (often messy)
    max_days: int = 120             # keep near-term options
    max_spread_pct: float = 0.20    # filter illiquid chains by bid/ask spread
    export_dir: Path = Path.home() / "Documents" / "CallBrief"
    export_filename: str | None = None  # if None, auto-named


# ----------------------------
# Helpers
# ----------------------------

def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def pct(a: float, b: float) -> float:
    """Return a/b as pct; handles divide-by-zero."""
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def days_to(expiration: str, today: pd.Timestamp) -> int:
    exp = pd.to_datetime(expiration)
    return int((exp - today).days)


def annualize(simple_return: float, dte: float) -> float:
    """Simple annualization using (1+r)^(365/dte)-1."""
    if dte <= 0 or np.isnan(simple_return):
        return np.nan
    return (1.0 + simple_return) ** (365.0 / dte) - 1.0


# ----------------------------
# Core
# ----------------------------

def fetch_calls(cfg: RunConfig) -> tuple[pd.DataFrame, dict]:
    tkr = yf.Ticker(cfg.ticker)

    # Underlying info
    info = {}
    try:
        info = tkr.fast_info if hasattr(tkr, "fast_info") else {}
    except Exception:
        info = {}

    # Get spot price from fastest reliable source
    spot = None
    for k in ["lastPrice", "last_price", "last", "regularMarketPrice"]:
        if isinstance(info, dict) and k in info:
            spot = safe_float(info.get(k))
            if not np.isnan(spot):
                break

    if spot is None or (isinstance(spot, float) and np.isnan(spot)):
        # fallback to history
        hist = tkr.history(period="5d")
        if hist.empty:
            raise RuntimeError(f"Could not fetch price history for {cfg.ticker}.")
        spot = float(hist["Close"].iloc[-1])

    today = pd.Timestamp(datetime.now().date())

    expirations = list(getattr(tkr, "options", []) or [])
    if not expirations:
        raise RuntimeError(f"No options expirations returned for {cfg.ticker}.")

    rows = []

    for exp in expirations:
        dte = days_to(exp, today)
        if dte < cfg.min_days or dte > cfg.max_days:
            continue

        try:
            chain = tkr.option_chain(exp)
        except Exception:
            continue

        calls = chain.calls.copy()
        if calls.empty:
            continue

        # Ensure numeric
        for col in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]:
            if col in calls.columns:
                calls[col] = pd.to_numeric(calls[col], errors="coerce")

        calls["expiration"] = pd.to_datetime(exp)
        calls["dte"] = dte
        calls["spot"] = spot

        # Mid + spreads
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
        calls["spread"] = calls["ask"] - calls["bid"]
        calls["spread_pct_mid"] = calls["spread"] / calls["mid"]

        # Basic payoff/return style metrics (simple, not “edge”, just useful)
        # Breakeven for long call: strike + premium (use mid)
        calls["breakeven"] = calls["strike"] + calls["mid"]

        # If this were a covered call, simple "premium/spot" return
        calls["covered_call_prem_pct"] = calls["mid"] / calls["spot"]
        calls["covered_call_prem_ann"] = calls.apply(
            lambda r: annualize(r["covered_call_prem_pct"], r["dte"]), axis=1
        )

        # “Moneyness” = spot/strike
        calls["moneyness"] = calls["spot"] / calls["strike"]

        # Filter by spread quality (remove super-wide)
        calls = calls[
            (calls["mid"] > 0)
            & (calls["spread_pct_mid"].isna() | (calls["spread_pct_mid"] <= cfg.max_spread_pct))
        ].copy()

        # Rank: higher OI + volume, tighter spreads
        # Simple composite score (feel free to change)
        calls["liq_score"] = (
            np.log1p(calls["openInterest"].fillna(0))
            + 0.5 * np.log1p(calls["volume"].fillna(0))
            - 2.0 * calls["spread_pct_mid"].fillna(0)
        )

        rows.append(calls)

    if not rows:
        raise RuntimeError(
            f"No calls found within {cfg.min_days}-{cfg.max_days} DTE for {cfg.ticker}."
        )

    df = pd.concat(rows, ignore_index=True)

    # Clean + sort
    keep_cols = [
        "contractSymbol", "expiration", "dte",
        "spot", "strike", "moneyness",
        "bid", "ask", "mid", "spread", "spread_pct_mid",
        "lastPrice", "impliedVolatility",
        "volume", "openInterest", "liq_score",
        "breakeven",
        "covered_call_prem_pct", "covered_call_prem_ann",
        "currency"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values(["expiration", "strike"]).reset_index(drop=True)

    meta = {
        "ticker": cfg.ticker.upper(),
        "spot": spot,
        "pulled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "min_days": cfg.min_days,
        "max_days": cfg.max_days,
        "max_spread_pct": cfg.max_spread_pct,
        "expirations_kept": int(df["expiration"].nunique()),
        "rows": int(len(df)),
    }
    return df, meta


def export_to_excel(df: pd.DataFrame, meta: dict, cfg: RunConfig) -> Path:
    cfg.export_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg.export_filename or f"{meta['ticker']}_calls_{ts}.xlsx"
    out_path = cfg.export_dir / fname

    # Helpful summary sheets
    summary = pd.DataFrame([meta])

    # Top liquid across all expirations
    top_liq = df.sort_values("liq_score", ascending=False).head(50)

    # A few “buckets” by moneyness (rough)
    def bucket(m):
        if pd.isna(m):
            return "unknown"
        if m >= 1.05:
            return "ITM"
        if m <= 0.95:
            return "OTM"
        return "ATM-ish"

    df2 = df.copy()
    df2["bucket"] = df2["moneyness"].apply(bucket)

    by_exp = (
        df2.groupby(["expiration", "bucket"], as_index=False)
           .agg(
               contracts=("contractSymbol", "count"),
               avg_iv=("impliedVolatility", "mean"),
               avg_spread_pct=("spread_pct_mid", "mean"),
               avg_oi=("openInterest", "mean"),
           )
           .sort_values(["expiration", "bucket"])
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Calls_All", index=False)
        top_liq.to_excel(writer, sheet_name="Top_Liquidity", index=False)
        by_exp.to_excel(writer, sheet_name="By_Exp_Bucket", index=False)

    return out_path


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    cfg = RunConfig(
        ticker="AAPL",      # <-- change me
        min_days=7,
        max_days=120,
        max_spread_pct=0.20,
    )

    df_calls, meta = fetch_calls(cfg)
    out = export_to_excel(df_calls, meta, cfg)

    print("Done.")
    print("Ticker:", meta["ticker"])
    print("Spot:", meta["spot"])
    print("Rows:", meta["rows"], "Expirations:", meta["expirations_kept"])
    print("Excel:", out)
