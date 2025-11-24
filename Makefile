# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# format code with black
format:
	black src/*.py

# Run linter (flake8 for Python files)
lint:
	flake8 --ignore=W503,C,N *.py

# Run the preprocess test
preprocess:
	python tests/preprocess_test.py

# run convert_xml bash script
convert:
	./scripts/convert_xml.sh

# run load data 
load:
	python tests/test_load_data.py

# train crf model
crf:
	./scripts/train_crf.sh

# train bilstm crf model 
bilstm:
	./scripts/train_bilstm_crf.sh

# Clean up Python cache files
clean:
	rm -rf __pycache__ .DS_Store

# Default target
all: install format lint run