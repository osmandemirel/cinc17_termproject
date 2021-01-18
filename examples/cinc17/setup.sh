#!/bin/bash

mkdir data && cd data

curl -O https://archive.physionet.org/challenge/2017/training2017.zip
unzip training2017.zip
curl -O https://archive.physionet.org/challenge/2017/sample2017.zip
unzip sample2017.zip
curl -O https://archive.physionet.org/challenge/2017/REFERENCE-v3.csv

cd ..

python build_datasets.py
