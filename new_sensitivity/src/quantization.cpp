#include "gradient/quantization.hpp"

namespace gradient {

std::vector<F> quantize_vector(const std::vector<double> &values) {
    std::vector<F> out;
    out.reserve(values.size());
    for (double v : values) {
        out.emplace_back(quantize_scalar(v));
    }
    return out;
}

std::vector<std::vector<F>> quantize_matrix(const std::vector<std::vector<double>> &matrix) {
    std::vector<std::vector<F>> out;
    out.reserve(matrix.size());
    for (const auto &row : matrix) {
        out.emplace_back(quantize_vector(row));
    }
    return out;
}

std::vector<double> dequantize_vector(const std::vector<F> &values, int depth) {
    std::vector<double> out;
    out.reserve(values.size());
    for (const auto &v : values) {
        out.emplace_back(dequantize_scalar(v, depth));
    }
    return out;
}

std::vector<std::vector<double>> dequantize_matrix(const std::vector<std::vector<F>> &matrix, int depth) {
    std::vector<std::vector<double>> out;
    out.reserve(matrix.size());
    for (const auto &row : matrix) {
        out.emplace_back(dequantize_vector(row, depth));
    }
    return out;
}

} // namespace gradient
