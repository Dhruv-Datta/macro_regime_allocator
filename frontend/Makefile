SHELL := /bin/bash
.PHONY: app install clean

# Launch the Streamlit dashboard
app:
	uv run streamlit run app.py

# Install frontend dependencies
install:
	uv sync

# Clean cache
clean:
	rm -rf __pycache__ .streamlit
