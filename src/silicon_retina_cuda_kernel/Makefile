all:
	nvcc -o main main.cpp cnpy.cpp sillicon_retina_kernel.cu `pkg-config --cflags --libs opencv4` -std=c++17 -lz

online:
	nvcc -o test_online main2.cpp cnpy.cpp sillicon_retina_kernel.cu `pkg-config --cflags --libs opencv4` -std=c++17 -lz