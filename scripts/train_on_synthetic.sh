#!/usr/bin/env bash

python ./tests/test_crf_on_synthetic.py > results/crf/synthetic_data_crf.txt
python ./tests/test_bilstm_on_synthetic.py > results/bilstm_crf/synthetic_data_bilstm.txt