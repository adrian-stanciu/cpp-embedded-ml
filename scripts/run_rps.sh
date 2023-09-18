#!/bin/sh

if [ $# -ne 1 ]; then
    echo "usage: $0 <input-image>"
    exit 1
fi

IMAGE="$1"

make run \
MODEL=models/rps_big_dataset_model.tflite \
LABELS=models/rps_labels.txt \
NUM_THREADS=4 \
IN_IMAGE="${IMAGE}"

