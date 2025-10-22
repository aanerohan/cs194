#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <openssl/sha.h>
#include "cpu_ref.hpp"

extern "C" void launch_softmax_kernel(const float* x_d, float* y_d, int N, int D, cudaStream_t stream);

static std::string sha256_hex(const std::vector<float>& data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char*)data.data(), data.size() * sizeof(float), hash);
    static const char* hexd = "0123456789abcdef";
    std::string out; out.resize(64);
    for (int i = 0; i < 32; ++i) { out[2*i] = hexd[(hash[i]>>4)&0xF]; out[2*i+1] = hexd[hash[i]&0xF]; }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s N D\n", argv[0]); return 1; }
    int N = std::atoi(argv[1]), D = std::atoi(argv[2]);

    const char* seed_env = std::getenv("TB_SEED");
    unsigned long long seed = seed_env ? std::strtoull(seed_env, nullptr, 10) : 12345ull;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> x(N*D);
    for (auto& v : x) v = dist(rng);

    float *x_d = nullptr, *y_d = nullptr;
    cudaMalloc(&x_d, N*D*sizeof(float));
    cudaMalloc(&y_d, N*D*sizeof(float));
    cudaMemcpy(x_d, x.data(), N*D*sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream; cudaStreamCreate(&stream);

    launch_softmax_kernel(x_d, y_d, N, D, stream);
    cudaStreamSynchronize(stream);

    int iters = 10;
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) launch_softmax_kernel(x_d, y_d, N, D, stream);
    cudaEventRecord(stop, stream); cudaEventSynchronize(stop);
    float ms_total = 0.0f; cudaEventElapsedTime(&ms_total, start, stop);
    float ms = ms_total / iters;

    std::vector<float> y(N*D);
    cudaMemcpy(y.data(), y_d, N*D*sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> y_ref(N*D);
    cpu_softmax(x.data(), y_ref.data(), N, D);

    double max_abs = 0.0, max_rel = 0.0, row_dev = 0.0;
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < D; ++j) {
            double a = (double)y[i*D+j], b = (double)y_ref[i*D+j];
            double absd = std::abs(a - b);
            max_abs = std::max(max_abs, absd);
            if (b != 0.0) max_rel = std::max(max_rel, absd / std::abs(b));
            sum += a;
        }
        row_dev = std::max(row_dev, std::abs(sum - 1.0));
    }
    bool ok = (row_dev <= 1e-6) && (max_abs <= 1e-6 || max_rel <= 1e-6);

    std::string checksum = sha256_hex(y);
    std::printf("{\"ok\":%s, \"ms\":%.4f, \"checksum\":\"%s\"}\n", ok ? "true" : "false", ms, checksum.c_str());

    cudaFree(x_d); cudaFree(y_d); cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
    return ok ? 0 : 2;
}
