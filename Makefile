.PHONY: setup install preprocess train run

install:
	pip install -r requirements.txt

preprocess:
	python prepro_tinyshakespeare.py
	python prepro_tinystories.py

train:
	go run ./cmd/traingpt2

test:
	go run ./cmd/testgpt2

setup: install preprocess

all: setup test