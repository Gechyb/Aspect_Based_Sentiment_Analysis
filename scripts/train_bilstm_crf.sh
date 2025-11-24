#!/usr/bin/env bash

# Run restuarants trainings
python -m src.train_bilstm_crf --domain restaurants > results/bilstm_crf/rest_dim200.txt
python -m src.train_bilstm_crf --domain restaurants --use_glove > results/bilstm_crf/rest_dim200_glov.txt # without gloVe embeddings

# Run laptops trainings
python -m src.train_bilstm_crf --domain laptops > results/bilstm_crf/lap_dim200.txt
python -m src.train_bilstm_crf --domain laptops --use_glove > results/bilstm_crf/lap_dim200_glov.txt # without gloVe embeddings