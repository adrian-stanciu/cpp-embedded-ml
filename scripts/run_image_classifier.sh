#!/bin/sh

if [ $# -ne 1 ]; then
    echo "usage: $0 <input-image>"
    exit 1
fi

IMAGE="$1"

make run \
MODEL=models/mobilenet_v1_1.0_224_quant.tflite \
LABELS=models/labels_mobilenet_quant_v1_224.txt \
NUM_THREADS=4 \
IN_IMAGE="${IMAGE}"

