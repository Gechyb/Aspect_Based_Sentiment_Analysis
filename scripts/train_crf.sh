#!/usr/bin/env bash
python -m src.train_crf --domain restaurants > results/crf/restaurants.txt
python -m src.train_crf --domain laptops > results/crf/laptops.txt

# with glove 
python -m src.train_crf --domain restaurants --use_glove > results/crf/restaurants_glove.txt
python -m src.train_crf --domain laptops --use_glove > results/crf/laptops_glove.txt
