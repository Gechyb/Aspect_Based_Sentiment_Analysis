#!/usr/bin/env bash
python -m src.train_crf --domain restaurants > results/crf/restaurants.txt
python -m src.train_crf --domain laptops > results/crf/laptops.txt

