all:
	nvcc -o test main.cpp cnpy.cpp sillicon_retina_kernel.cu `pkg-config --cflags --libs opencv4` -std=c++17 -lz