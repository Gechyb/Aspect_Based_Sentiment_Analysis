# Aspect-Based Sentiment Analysis via Sequence Tagging

**Authors:** Ogechukwu Ezenwa, Supriya Nannapaneni, Sejal Jagtap
**Course:** ECE-684 Natural Language Processing (Final Project), Duke University  
**Date:** Fall 2025


## Project Overview

This project implements a unified approach to Aspect-Based Sentiment Analysis (ABSA) using sequence tagging. Unlike traditional sentiment analysis that provides document-level sentiment, our system jointly extracts aspect terms and their associated sentiments (positive, negative, neutral) within a single model.

### Key Features
- **Joint Extraction**: Simultaneous aspect detection and sentiment classification
- **Two Model Implementations**: CRF baseline and BiLSTM-CRF with pre-trained embeddings
- **BIO-Sentiment Tagging**: Extended tagging scheme (B-POS, I-POS, B-NEG, I-NEG, B-NEU, I-NEU, O)

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone git@github.com:Gechyb/Aspect_Based_Sentiment_Analysis.git
cd Aspect_Based_Sentiment_Analysis
```

2. **Create a virtual environment:**
```bash
# Using conda (recommended)
conda create -n nlp_project python=3.12
conda activate nlp_project

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Download Data and Embeddings

1. **Dataset**: [SemEval 2014 Task 4](https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis/data)
    - The raw files downloaded from kaggle should be in `data/raw` folder.
        - You must preprocess the XML files, not the CSV. 
        - The CSV files do NOT contain the aspect terms.
        - The XML files DO contain the aspect terms and sentiment labels.
    - The preprocessed data should be in `data/intermediate` folder.
        - Files: `restaurants.jsonl`, `laptops.jsonl`
        - converted the xml to jsonl for training the models.
```bash
make convert
```

2. **GloVe Embeddings** (Required for best performance):
```bash
mkdir -p data/glove
cd data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../..
```

## Usage

### Training Models

#### 1. CRF Baseline
```bash
# Restaurants domain
python -m src.train_crf --domain restaurants

# Laptops domain
python -m src.train_crf --domain laptops
```

#### 2. BiLSTM-CRF (without pre-trained embeddings)
```bash
python -m src.train_bilstm_crf --domain restaurants
```

#### 3. BiLSTM-CRF + GloVe (best performance)
```bash
# Restaurants domain
python -m src.train_bilstm_crf --domain restaurants --use_glove

# Laptops domain
python -m src.train_bilstm_crf --domain laptops --use_glove

# Custom GloVe directory
python -m src.train_bilstm_crf --domain restaurants --use_glove --glove_dir /path/to/glove
```

### Running Scripts

The `scripts/` folder contains convenient bash scripts:

```bash
# Convert XML to JSON (if needed)
bash scripts/convert_xml.sh

# Train all models
bash scripts/train_crf.sh
bash scripts/train_bilstm_crf.sh

# Run crf on synthetic data
python -m scripts.run_crf_synthetic
```

## Project Structure

```
Aspect_Based_Sentiment_Analysis/
├── data/
│   ├── glove/                      # GloVe embeddings (download separately)
│   │   ├── glove.6B.50d.txt
│   │   ├── glove.6B.100d.txt
│   │   ├── glove.6B.200d.txt
│   │   └── glove.6B.300d.txt
│   ├── intermediate/               # Preprocessed data
│   ├── raw/                        # Original XML files
│   ├── restaurants.jsonl          # Preprocessed restaurant reviews
│   └── laptops.jsonl              # Preprocessed laptop reviews
│
├── results/
│   ├── bilstm_crf/                # BiLSTM-CRF outputs
│   └── crf/                       # CRF baseline outputs
│       ├── restaurants.txt
│       └── laptops.txt
│
├── scripts/
│   ├── convert_xml.sh             # Data preprocessing
│   ├── train_crf.sh               # Train CRF baseline
│   └──  train_bilstm_crf.sh        # Train BiLSTM-CRF
│   
├── src/
│   ├── models/
│   │   |── bilstm_crf.py          # BiLSTM-CRF architecture
|   |   └── crf_model.py           # CRF architecture
|   |
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters
│   ├── create_jsonl_from_xml.py   # Data preprocessing
│   ├── data_utils.py              # Data loading utilities
│   ├── features_crf.py            # CRF feature engineering
│   ├── glove_utils.py             # GloVe embedding loader
│   ├── metrics.py                 # Evaluation metrics
│   ├── preprocess.py              # Text preprocessing
│   ├── tagging_scheme.py          # BIO-Sentiment tags
│   ├── train_bilstm_crf.py        # BiLSTM-CRF training
│   └── train_crf.py               # CRF baseline training
│
├── tests/
│   ├── preprocess_test.py
│   ├── test_crf_synthetic.py
│   └── test_load_data.py
│
├── .gitignore
├── Makefile
├── README.md                       # Setup instructions
├── requirements.txt
```

## Configuration

### Hyperparameters

Edit `src/config.py` to tune model parameters:

```python
class Config:
    # Model architecture
    embedding_dim = 100    # GloVe dimension (50, 100, 200, or 300)
    hidden_dim = 128       # LSTM hidden size
    dropout = 0.3          # Dropout rate
    
    # Training
    batch_size = 32
    epochs = 10
    lr = 0.001             # Learning rate
    seed = 42              # Random seed for reproducibility
```

### Recommended Settings for Best Performance

```python
# For optimal results
embedding_dim = 200
hidden_dim = 256
lr = 0.0005
epochs = 20
```

## Testing

Run unit tests:

```bash
# Test preprocessing
python -m tests.preprocess_test

# Test data loading
python -m tests.test_load_data

```
## Evaluation Metrics

Models are evaluated using:
- **Span-level F1 Score**: Measures exact aspect span and sentiment detection
- **Precision**: Accuracy of predicted aspects
- **Recall**: Coverage of gold aspects

Evaluation uses the `seqeval` library for sequence labeling metrics.

---

## Tagging Scheme

We use a BIO-Sentiment tagging scheme:

| Tag | Meaning |
|-----|---------|
| `O` | Outside any aspect |
| `B-POS` | Beginning of positive aspect |
| `I-POS` | Inside positive aspect |
| `B-NEG` | Beginning of negative aspect |
| `I-NEG` | Inside negative aspect |
| `B-NEU` | Beginning of neutral aspect |
| `I-NEU` | Inside neutral aspect |
| `PAD` | Padding token (ignored) |

## References

1. **SemEval 2014 Task 4**: Aspect Based Sentiment Analysis
   - Pontiki et al. (2014)

2. **BiLSTM-CRF for Sequence Tagging**
   - Lample et al. (2016): Neural Architectures for Named Entity Recognition

3. **GloVe Embeddings**
   - Pennington et al. (2014): GloVe: Global Vectors for Word Representation

4. **Seqeval Library**
   - https://github.com/chakki-works/seqeval

---
