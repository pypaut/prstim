SRC=src/main.py


all: lint
	python3 src/main.py

lint:
	black -l 79 ${SRC}
	flake8 ${SRC}
