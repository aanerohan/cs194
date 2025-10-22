#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

inline void cpu_softmax(const float* x, float* y, int N, int D) {
    for (int i = 0; i < N; ++i) {
        const float* row = x + i * D;
        float* out = y + i * D;
        float mx = row[0];
        for (int j = 1; j < D; ++j) mx = std::max(mx, row[j]);
        double sum = 0.0;
        for (int j = 0; j < D; ++j) {
            double v = std::exp(double(row[j] - mx));
            out[j] = float(v);
            sum += v;
        }
        double inv = 1.0 / sum;
        for (int j = 0; j < D; ++j) out[j] = float(double(out[j]) * inv);
    }
}
