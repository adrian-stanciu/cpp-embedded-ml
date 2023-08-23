### C++ image classifier with TensorFlow Lite on Raspberry Pi

#### Prerequisites
* hardware:
    * Raspberry Pi 4 model B
    * 8 GB microSD card
    * Raspberry Pi Camera module 3
* software:
    * 64-bit Raspberry Pi OS
    * g++ (with c++20 support)
    * make
    * libpthread
    * libcamera
    * libgstreamer

#### Install
    sh install.sh

#### Compile
    make build [-j <number-of-jobs>]

#### Run
    make run \
        [LABELS=<path-to-labels>] \
        [MODEL=<path-to-model>] \
        [IN_IMAGE=<path-to-input-image>] \
        [OUT_IMAGE=<path-to-output-image>] \
        [NUM_THREADS=<number-of-threads>] \
        [PLAY_RPS=YES]

#### Clean up
    make clean

#### Uninstall
    sh uninstall.sh

