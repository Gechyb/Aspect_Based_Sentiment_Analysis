# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# format code with black
format:
	black *.py

# Run linter (flake8 for Python files)
lint:
	flake8 --ignore=W503,C,N *.py

# Run the data_analysis.py
preprocess:
	python preprocess.py

# run convert_xml bash script
convert:
	./scripts/convert_xml.sh
# Clean up Python cache files
clean:
	rm -rf __pycache__ .DS_Store

# Default target
all: install format lint run