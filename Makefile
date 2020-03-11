.PHONY: all lint test license FORCE

all: test

license: FORCE
	python scripts/update_headers.py

lint: FORCE
	flake8

test: lint FORCE
	python exampes/model1 -n 2 -s 2

FORCE:
