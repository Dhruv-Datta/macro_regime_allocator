"""Copy data files from backend outputs to frontend public/data for serving."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "macro_regime_allocator"
DEST = Path(__file__).resolve().parent.parent / "public" / "data"
DEST.mkdir(parents=True, exist_ok=True)

files = [
    (ROOT / "outputs" / "backtest_results.csv", "backtest.csv"),
    (ROOT / "outputs" / "investment_metrics.csv", "metrics.csv"),
    (ROOT / "data" / "features.csv", "features.csv"),
    (ROOT / "config.yaml", "config.yaml"),
    (ROOT / "outputs" / "report.md", "report.md"),
]

for src, name in files:
    if src.exists():
        shutil.copy2(src, DEST / name)
        print(f"  copied {src.name} -> public/data/{name}")
    else:
        print(f"  MISSING {src}")

print("Done.")
