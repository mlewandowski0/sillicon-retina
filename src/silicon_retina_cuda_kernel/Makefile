# Compile for the GPU in this machine; override e.g. `make ARCH=-arch=sm_86`.
# (Native SASS avoids "PTX compiled with an unsupported toolchain" errors when
# the CUDA toolkit is newer than the driver.)
ARCH ?= -arch=native

all:
	nvcc $(ARCH) -o main main.cpp cnpy.cpp sillicon_retina_kernel.cu `pkg-config --cflags --libs opencv4` -std=c++17 -lz

online:
	nvcc $(ARCH) -o main_online main2.cpp cnpy.cpp sillicon_retina_kernel.cu `pkg-config --cflags --libs opencv4` -std=c++17 -lz
