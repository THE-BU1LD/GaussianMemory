#!/usr/bin/env bash
set -e
python train.py --benchmark associative_recall --model gmf
python train.py --benchmark associative_recall --model gru
python train.py --benchmark associative_recall --model transformer
python train.py --benchmark associative_recall --model mlp
