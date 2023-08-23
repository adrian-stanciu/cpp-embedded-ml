.PHONY: build run zip clean

CC := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror $(shell pkg-config --cflags opencv4)
LDLIBS := -ltensorflow-lite -lfmt $(shell pkg-config --libs opencv4)

SRCS := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp,%.o,${SRCS})
HDRS := $(wildcard *.hpp)
SCRIPTS := $(wildcard *.sh)

build: image_classifier
image_classifier: ${OBJS}
cmdline_parser.o: cmdline_parser.cpp cmdline_parser.hpp
image_classifier.o: image_classifier.cpp image_classifier.hpp
main.o: main.cpp ${HDRS}
rps.o: rps.cpp rps.hpp

LABELS ?= labels_mobilenet_quant_v1_224.txt
MODEL ?= mobilenet_v1_1.0_224_quant.tflite
RUN_CMD := LC_ALL=C LIBCAMERA_LOG_LEVELS=ERROR ./image_classifier -l ${LABELS} -m ${MODEL}

ifdef IN_IMAGE
RUN_CMD := ${RUN_CMD} -i ${IN_IMAGE}
endif

ifdef OUT_IMAGE
RUN_CMD := ${RUN_CMD} -o ${OUT_IMAGE}
endif

ifdef NUM_THREADS
RUN_CMD := ${RUN_CMD} -t ${NUM_THREADS}
endif

ifeq (${PLAY_RPS}, YES)
RUN_CMD := ${RUN_CMD} -g
endif

run: build
	${RUN_CMD}

zip: ${SRCS} ${HDRS} ${SCRIPTS} Makefile README.md
	zip image_classifier.zip $^

clean:
	rm -f *.o image_classifier image_classifier.zip

