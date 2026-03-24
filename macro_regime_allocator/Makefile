SHELL := /bin/bash
.PHONY: run predict signal fast install clean

# Full run: download data + backtest + evaluate + plots
run:
	source .env.local && export FRED_API_KEY && uv run python main.py

# Full run + show latest predicted allocation (downloads fresh data)
predict:
	source .env.local && export FRED_API_KEY && uv run python main.py --predict-latest

# Quick signal: what should I do this month? (uses cached data)
signal:
	source .env.local && export FRED_API_KEY && uv run python main.py --skip-download --predict-latest

# Quick re-run using cached data (no download)
fast:
	source .env.local && export FRED_API_KEY && uv run python main.py --skip-download

# Install dependencies
install:
	uv sync

# Remove generated outputs
clean:
	rm -rf outputs/*.csv outputs/*.md outputs/*.joblib outputs/plots/*.png
