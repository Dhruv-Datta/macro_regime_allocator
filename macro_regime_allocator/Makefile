SHELL := /bin/bash
.PHONY: run predict fast rerun rerun-predict install clean

# Full run: download data + backtest + evaluate + plots
run:
	source .env.local && export FRED_API_KEY &&uv run python main.py

# Full run + show latest predicted allocation
predict:
	source .env.local && export FRED_API_KEY &&uv run python main.py --predict-latest

# Quick re-run using cached data (no download)
fast:
	source .env.local && export FRED_API_KEY &&uv run python main.py --skip-download

# Incremental model with rolling window (cached data)
rerun:
	source .env.local && export FRED_API_KEY &&uv run python main.py --skip-download --model incremental --window rolling

# Incremental + rolling + latest prediction
rerun-predict:
	source .env.local && export FRED_API_KEY &&uv run python main.py --skip-download --model incremental --window rolling --predict-latest

# Install dependencies
install:
	uv pip install -r requirements.txt

# Remove generated outputs
clean:
	rm -rf outputs/*.csv outputs/*.joblib outputs/plots/*.png
