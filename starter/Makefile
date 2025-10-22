CUDA_ARCH ?= native
NVCC ?= nvcc

all: softmax_bench naive_softmax_bench

softmax_bench: main.cu kernel.cu cpu_ref.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -o $@ main.cu kernel.cu

naive_softmax_bench: main.cu naive_softmax.cu cpu_ref.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -o $@ main.cu naive_softmax.cu

run: softmax_bench
	./softmax_bench 2048 1024

baseline: naive_softmax_bench
	./naive_softmax_bench 2048 1024
