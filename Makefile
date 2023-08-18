.PHONY: build run zip clean

CC := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror $(shell pkg-config --cflags opencv4)
LDLIBS := -ltensorflow-lite -lfmt $(shell pkg-config --libs opencv4)

build: image_classifier
image_classifier: image_classifier.o ml.o util.o
image_classifier.o: image_classifier.cpp camera.hpp ml.hpp util.hpp
ml.o: ml.cpp ml.hpp
util.o: util.cpp util.hpp

LABELS ?= labels_mobilenet_quant_v1_224.txt
MODEL ?= mobilenet_v1_1.0_224_quant.tflite
RUN_CMD := LC_ALL=C LIBCAMERA_LOG_LEVELS=ERROR ./image_classifier -l ${LABELS} -m ${MODEL}
run: build
ifdef IMAGE
	${RUN_CMD} -i ${IMAGE}
else
	${RUN_CMD}
endif

SRCS := $(wildcard *.cpp)
HDRS := $(wildcard *.hpp)
SCRIPTS := $(wildcard *.sh)
zip: ${SRCS} ${HDRS} ${SCRIPTS} Makefile README.md
	zip image_classifier.zip $^

clean:
	rm -f *.o image_classifier image_classifier.zip

