#!/usr/bin/env bash

echo "Converting Restaurants..."
python src/create_jsonl_from_xml.py \
    --xml data/raw/Restaurants_Train_v2.xml \
    --out data/intermediate/restaurants.jsonl


echo "Converting Laptops..."
python src/create_jsonl_from_xml.py \
    --xml data/raw/Laptop_Train_v2.xml \
    --out data/intermediate/laptops.jsonl

echo "Done."