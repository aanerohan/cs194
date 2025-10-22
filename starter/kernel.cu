#include <cuda_runtime.h>

__global__ void row_softmax(const float* __restrict__ x, float* __restrict__ y, int N, int D) {
    int i = blockIdx.x; // one block per row
    if (i >= N) return;

    extern __shared__ float smem[]; // optional

    // Compute row max (naive serial by thread 0 for now; contestants can optimize)
    __shared__ float smax;
    if (threadIdx.x == 0) {
        float m = x[i*D];
        for (int j = 1; j < D; ++j) m = fmaxf(m, x[i*D+j]);
        smax = m;
    }
    __syncthreads();

    // Compute exp and partial sums
    float local = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float e = __expf(x[i*D+j] - smax);
        y[i*D+j] = e;
        local += e;
    }

    // Reduce local sums
    __shared__ float red[1024];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = red[0];
    __syncthreads();

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        y[i*D+j] = y[i*D+j] / ssum;
    }
}

extern "C" void launch_softmax_kernel(const float* x_d, float* y_d, int N, int D, cudaStream_t stream) {
    dim3 grid(N), block(256);
    row_softmax<<<grid, block, 0, stream>>>(x_d, y_d, N, D);
}
