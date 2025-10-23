#pragma once

#include "config_pc.hpp"
#include "quantization.h"

#include <vector>

namespace gradient {

inline F quantize_scalar(double value) {
    return quantize(static_cast<float>(value));
}

inline double dequantize_scalar(const F &value, int depth = 1) {
    return static_cast<double>(dequantize(value, depth));
}

std::vector<F> quantize_vector(const std::vector<double> &values);
std::vector<std::vector<F>> quantize_matrix(const std::vector<std::vector<double>> &matrix);

std::vector<double> dequantize_vector(const std::vector<F> &values, int depth = 1);
std::vector<std::vector<double>> dequantize_matrix(const std::vector<std::vector<F>> &matrix, int depth = 1);

} // namespace gradient
