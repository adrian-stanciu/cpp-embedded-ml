#!/bin/sh

sudo apt remove -y tensorflow-lite
sudo apt remove -y libopencv-dev
sudo apt remove -y libfmt-dev

rm -f labels_mobilenet_quant_v1_224.txt
rm -f mobilenet_v1_1.0_224_quant.tflite

