#!/bin/sh

make run \
MODEL=models/rps_big_dataset_model.tflite \
LABELS=models/rps_labels.txt \
NUM_THREADS=4 \
PLAY_RPS=YES

