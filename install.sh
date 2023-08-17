#!/bin/sh

set -e

wget https://github.com/prepkg/tensorflow-lite-raspberrypi/releases/latest/download/tensorflow-lite_64.deb
sudo apt install -y ./tensorflow-lite_64.deb
rm -f tensorflow-lite_64.deb

sudo apt install -y libopencv-dev
sudo apt install -y libfmt-dev

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip
unzip -j mobilenet_v1_1.0_224_quant_and_labels.zip labels_mobilenet_quant_v1_224.txt
unzip -j mobilenet_v1_1.0_224_quant_and_labels.zip mobilenet_v1_1.0_224_quant.tflite
rm -f mobilenet_v1_1.0_224_quant_and_labels.zip

