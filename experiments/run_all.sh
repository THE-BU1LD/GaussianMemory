#!/usr/bin/env bash
set -e
python train.py --benchmark associative_recall --model gmf
python train.py --benchmark needle_in_haystack --model gmf
python train.py --benchmark temporal_prediction --model gmf
python train.py --benchmark noisy_retrieval --model gmf
python eval.py --benchmark associative_recall --model gmf
python eval.py --benchmark needle_in_haystack --model gmf
python eval.py --benchmark temporal_prediction --model gmf
python eval.py --benchmark noisy_retrieval --model gmf
