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

    raw = yf.download(tickers, start=cfg.start_date, end=cfg.end_date,
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

    # VIX
    try:
        vix = yf.download(cfg.vix_ticker, start=cfg.start_date,
                          end=cfg.end_date, auto_adjust=True, progress=False)
        vix_monthly = vix["Close"].resample("ME").last()
        vix_data["vix"] = vix_monthly
        print(f"  Downloaded VIX: {len(vix_monthly)} monthly obs")
    except Exception as e:
        print(f"  WARNING: Could not download VIX: {e}")

    # VIX3M (3-month VIX for term structure)
    try:
        vix3m = yf.download(cfg.vix3m_ticker, start=cfg.start_date,
                            end=cfg.end_date, auto_adjust=True, progress=False)
        vix3m_monthly = vix3m["Close"].resample("ME").last()
        vix_data["vix3m"] = vix3m_monthly
        print(f"  Downloaded VIX3M: {len(vix3m_monthly)} monthly obs")
    except Exception as e:
        print(f"  WARNING: Could not download VIX3M: {e}")

    vix_data.index.name = "date"
    return vix_data


def download_fred_data(cfg: Config) -> pd.DataFrame:
    """Download macro series from FRED."""
    from fredapi import Fred

    if not cfg.fred_api_key:
        raise ValueError("FRED_API_KEY not set.")

    fred = Fred(api_key=cfg.fred_api_key)
    series_dict = {}

    for name, series_id in cfg.fred_series.items():
        try:
            s = fred.get_series(series_id, observation_start=cfg.start_date,
                                observation_end=cfg.end_date)
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
