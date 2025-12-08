#!/usr/bin/env bash


python -m src.train_bilstm_crf --domain restaurants > results/bilstm_crf/rest_dim200.txt
python -m src.train_bilstm_crf --domain laptops > results/bilstm_crf/lap_dim200.txt

# with glove embeddings
python -m src.train_bilstm_crf --domain restaurants --use_glove > results/bilstm_crf/rest_dim200_glov.txt
python -m src.train_bilstm_crf --domain laptops --use_glove > results/bilstm_crf/lap_dim200_glov.txt