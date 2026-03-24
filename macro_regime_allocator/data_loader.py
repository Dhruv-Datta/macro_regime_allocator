"""
Download and clean macro + market data, align to monthly frequency.
"""

import os
import pandas as pd
import numpy as np
from config import Config


def download_asset_prices(cfg: Config) -> pd.DataFrame:
    """Download monthly adjusted close for equity proxy."""
    import yfinance as yf

    tickers = list(cfg.asset_tickers.values())
    names = list(cfg.asset_tickers.keys())

    # Extra lookback so rolling/YoY features have values from start_date
    lookback_start = (pd.Timestamp(cfg.start_date) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=lookback_start, end=cfg.end_date,
                      auto_adjust=True, progress=False)

    if len(tickers) == 1:
        prices = raw[["Close"]].copy()
        prices.columns = [names[0]]
    else:
        prices = raw["Close"].copy()
        rename_map = {v: k for k, v in cfg.asset_tickers.items()}
        prices.rename(columns=rename_map, inplace=True)

    prices = prices.resample("ME").last()
    prices.index.name = "date"
    return prices


def download_vix_data(cfg: Config) -> pd.DataFrame:
    """Download VIX and VIX3M (3-month VIX) for term structure."""
    import yfinance as yf

    vix_data = pd.DataFrame()

    lookback_start = (pd.Timestamp(cfg.start_date) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")

    # VIX
    try:
        vix = yf.download(cfg.vix_ticker, start=lookback_start,
                          end=cfg.end_date, auto_adjust=True, progress=False)
        vix_monthly = vix["Close"].resample("ME").last()
        vix_data["vix"] = vix_monthly
        print(f"  Downloaded VIX: {len(vix_monthly)} monthly obs")
    except Exception as e:
        print(f"  WARNING: Could not download VIX: {e}")

    # VIX3M (3-month VIX for term structure)
    try:
        vix3m = yf.download(cfg.vix3m_ticker, start=lookback_start,
                            end=cfg.end_date, auto_adjust=True, progress=False)
        vix3m_monthly = vix3m["Close"].resample("ME").last()
        vix_data["vix3m"] = vix3m_monthly
        print(f"  Downloaded VIX3M: {len(vix3m_monthly)} monthly obs")
    except Exception as e:
        print(f"  WARNING: Could not download VIX3M: {e}")

    vix_data.index.name = "date"
    return vix_data



def download_fred_data(cfg: Config) -> pd.DataFrame:
    """Download macro series from FRED.

    Uses ALFRED first-release data for revisable series (CPI, unemployment, etc.)
    to avoid lookahead bias from data revisions. Market-based series (fed funds,
    credit spreads, treasuries) use standard download since they aren't revised.
    """
    from fredapi import Fred

    if not cfg.fred_api_key:
        raise ValueError("FRED_API_KEY not set.")

    fred = Fred(api_key=cfg.fred_api_key)
    series_dict = {}
    # Pull extra history so YoY and rolling features can be computed
    # from the actual start_date without NaNs
    start = pd.Timestamp(cfg.start_date) - pd.DateOffset(months=18)
    end = pd.Timestamp(cfg.end_date)

    for name, series_id in cfg.fred_series.items():
        try:
            if name in cfg.fred_revisable_series:
                # ALFRED first-release: what was published initially, no revisions
                s = fred.get_series_first_release(series_id)
                s.index = pd.to_datetime(s.index)
                s = s[(s.index >= start) & (s.index <= end)]
                s = pd.to_numeric(s, errors="coerce")
                series_dict[name] = s
                print(f"  Downloaded {name} ({series_id}): {len(s)} obs [first-release]")
            else:
                s = fred.get_series(series_id, observation_start=start,
                                    observation_end=end)
                series_dict[name] = s
                print(f"  Downloaded {name} ({series_id}): {len(s)} obs")
        except Exception as e:
            print(f"  WARNING: Could not download {name} ({series_id}): {e}")

    if not series_dict:
        raise RuntimeError("No FRED series downloaded successfully.")

    macro = pd.DataFrame(series_dict)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.resample("ME").last()
    macro.index.name = "date"
    return macro


def load_data(cfg: Config) -> pd.DataFrame:
    """Full pipeline: download, merge, clean, save."""
    print("Downloading asset prices...")
    prices = download_asset_prices(cfg)
    print(f"  Prices shape: {prices.shape}")

    print("Downloading VIX data...")
    vix = download_vix_data(cfg)
    print(f"  VIX shape: {vix.shape}")

    print("Downloading FRED macro data...")
    macro = download_fred_data(cfg)
    print(f"  Macro shape: {macro.shape}")

    print("Merging and cleaning...")
    merged = prices.join(vix, how="outer").join(macro, how="outer")
    merged = merged.ffill()
    merged = merged.dropna(subset=["equity"], how="all")
    print(f"  Merged shape: {merged.shape}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    path = os.path.join(cfg.data_dir, "merged_monthly.csv")
    merged.to_csv(path)
    print(f"  Saved to {path}")

    return merged


if __name__ == "__main__":
    cfg = Config()
    df = load_data(cfg)
    print(df.tail())
