################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Edited by Marcos Luciano
# https://www.github.com/marcoslucianops
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
        $(error "CUDA_VER is not set")
endif

OPENCV?=
ifeq ($(OPENCV),)
        OPENCV=0
endif

CC := g++
NVCC := /usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS := -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I/opt/nvidia/deepstream/deepstream-7.0/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include -fPIC

ifeq ($(OPENCV), 1)
        COMMON = -DOPENCV
        CFLAGS += $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
        LIBS += $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

CUFLAGS := -I/opt/nvidia/deepstream/deepstream-7.0/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

LIBS += -fPIC -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS := -fPIC -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS := $(wildcard *.h)

SRCFILES := $(filter-out calibrator.cpp, $(wildcard *.cpp))

ifeq ($(OPENCV), 1)
        SRCFILES += calibrator.cpp
endif

SRCFILES += $(wildcard layers/*.cpp)
SRCFILES += $(wildcard *.cu)

TARGET_LIB := libnvdsinfer_custom_impl_Yolo.so

TARGET_OBJS := $(SRCFILES:.cpp=.o)
TARGET_OBJS := $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) $(COMMON) $(CFLAGS) -c -o $< $@

%.o: %.cu $(INCS) Makefile
	$(NVCC) $(CUFLAGS) --compiler-options '-fPIC' -c $< -o $@

$(TARGET_LIB): $(TARGET_OBJS)
	$(CC) $(TARGET_OBJS) $(LFLAGS) -o $@

clean:
	rm -rf $(TARGET_LIB) *.o
	$(info Build completed!)

