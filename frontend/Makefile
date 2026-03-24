.PHONY: install dev build clean prepare

install:
	npm install

prepare:
	python3 scripts/prepare_data.py

dev: prepare
	npm run dev

build: prepare
	npm run build

clean:
	rm -rf node_modules dist public/data
