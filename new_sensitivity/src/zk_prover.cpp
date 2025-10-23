#include "gradient/zk_prover.hpp"

#include "GKR.h"
#include "gradient/quantization.hpp"
#include "utils.hpp"

#include <cmath>
#include <stdexcept>

namespace gradient {
namespace {

using ::proof;
using ::quadratic_poly;
using ::linear_poly;

int ceil_log2(std::size_t n) {
    if (n <= 1) {
        return 0;
    }
    std::size_t value = 1;
    int bits = 0;
    while (value < n) {
        value <<= 1U;
        ++bits;
    }
    return bits;
}

std::size_t next_power_of_two(std::size_t n) {
    if (n <= 1) {
        return 1;
    }
    return static_cast<std::size_t>(1) << ceil_log2(n);
}

std::vector<std::vector<F>> pad_matrix(const std::vector<std::vector<F>> &matrix,
                                       std::size_t rows_pad,
                                       std::size_t cols_pad) {
    std::vector<std::vector<F>> padded(rows_pad, std::vector<F>(cols_pad, F_ZERO));
    for (std::size_t r = 0; r < matrix.size(); ++r) {
        for (std::size_t c = 0; c < matrix[r].size(); ++c) {
            padded[r][c] = matrix[r][c];
        }
    }
    return padded;
}

std::vector<F> pad_vector(const std::vector<F> &vec, std::size_t size_pad) {
    std::vector<F> padded(size_pad, F_ZERO);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        padded[i] = vec[i];
    }
    return padded;
}

std::vector<std::vector<F>> vector_to_matrix(const std::vector<F> &vec) {
    std::vector<std::vector<F>> matrix(vec.size(), std::vector<F>(1, F_ZERO));
    for (std::size_t i = 0; i < vec.size(); ++i) {
        matrix[i][0] = vec[i];
    }
    return matrix;
}

std::vector<std::vector<F>> transpose_matrix(const std::vector<std::vector<F>> &matrix) {
    if (matrix.empty()) {
        return {};
    }
    std::vector<std::vector<F>> transposed(matrix[0].size(), std::vector<F>(matrix.size(), F_ZERO));
    for (std::size_t r = 0; r < matrix.size(); ++r) {
        for (std::size_t c = 0; c < matrix[r].size(); ++c) {
            transposed[c][r] = matrix[r][c];
        }
    }
    return transposed;
}

std::vector<std::vector<F>> prepare_matrixes_local(const std::vector<std::vector<F>> &M1,
                                                   const std::vector<std::vector<F>> &M2,
                                                   const std::vector<F> &r1,
                                                   const std::vector<F> &r2) {
    std::vector<std::vector<F>> V;
    V.push_back(prepare_matrix(transpose_matrix(M1), r1));
    V.push_back(prepare_matrix(transpose_matrix(M2), r2));
    return V;
}

std::vector<F> random_field_vector(std::size_t length) {
    std::vector<F> values(length);
    for (std::size_t i = 0; i < length; ++i) {
        values[i] = F::random();
    }
    return values;
}

proof generate_2product_sumcheck(const std::vector<F> &input_v1,
                                 const std::vector<F> &input_v2) {
    std::vector<F> v1 = input_v1;
    std::vector<F> v2 = input_v2;
    const int rounds = static_cast<int>(std::log2(v1.size()));

    std::vector<F> randomness;
    std::vector<quadratic_poly> polys;

    F rand = F::random();
    for (int round = 0; round < rounds; ++round) {
        quadratic_poly poly = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
        quadratic_poly temp_poly = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
        int L = 1 << (rounds - 1 - round);
        for (int j = 0; j < L; ++j) {
            linear_poly l1(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
            linear_poly l2(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
            temp_poly = l1 * l2;
            poly = poly + temp_poly;

            v1[j] = v1[2 * j] + rand * (v1[2 * j + 1] - v1[2 * j]);
            v2[j] = v2[2 * j] + rand * (v2[2 * j + 1] - v2[2 * j]);
        }
        randomness.push_back(rand);
        polys.push_back(poly);
        rand = F::random();
    }

    proof Pr{};
    Pr.q_poly = polys;
    Pr.randomness.push_back(randomness);
    Pr.vr.push_back(v1[0]);
    Pr.vr.push_back(v2[0]);
    Pr.type = MATMUL_PROOF;
    return Pr;
}

proof prove_matrix_vector_product(std::vector<std::vector<F>> matrix,
                                  std::vector<std::vector<F>> vector_matrix,
                                  const std::vector<F> &result_vector) {
    const std::size_t row_bits = static_cast<std::size_t>(std::log2(matrix.size()));
    const std::size_t col_bits = static_cast<std::size_t>(std::log2(vector_matrix.size()));

    std::vector<F> r = random_field_vector(row_bits + col_bits);
    std::vector<F> r2(col_bits, F_ZERO);
    std::vector<F> r1(row_bits, F_ZERO);
    for (std::size_t i = 0; i < col_bits; ++i) {
        r2[i] = r[i];
    }
    for (std::size_t i = 0; i < row_bits; ++i) {
        r1[i] = r[col_bits + i];
    }

    auto prepared = prepare_matrixes_local(matrix, vector_matrix, r1, r2);
    proof Pr = generate_2product_sumcheck(prepared[0], prepared[1]);
    Pr.randomness.push_back(r1);
    Pr.randomness.push_back(r2);

    return Pr;
}

std::vector<proof> prove_dense_layer_outputs(const DenseLayer &layer,
                                             const DenseLayerCache &cache,
                                             const LayerSensitivity &sensitivity,
                                             const std::vector<GradientState> &next_layer_states) {
    const std::size_t outputs = layer.config.out_features;
    const std::size_t inputs = layer.config.in_features;
    std::vector<proof> proofs;
    proofs.reserve(outputs);

    for (std::size_t out_idx = 0; out_idx < outputs; ++out_idx) {
        std::vector<double> grad_out_raw(outputs, 0.0);
        if (next_layer_states.empty()) {
            grad_out_raw[out_idx] = 1.0;
        } else {
            const auto &state = std::holds_alternative<GradientVector>(next_layer_states[out_idx])
                                     ? std::get<GradientVector>(next_layer_states[out_idx])
                                     : throw std::runtime_error("Non-vector state in dense layer proof");
            grad_out_raw = state;
        }

        std::vector<double> mask = cache.mask;
        if (mask.empty()) {
            mask.assign(layer.config.out_features, 1.0);
        }

        std::vector<double> grad_out_masked(outputs, 0.0);
        for (std::size_t i = 0; i < outputs; ++i) {
            grad_out_masked[i] = grad_out_raw[i] * mask[i];
        }

        const auto &grad_in_state = sensitivity.gradients_per_output[out_idx];
        if (!std::holds_alternative<GradientVector>(grad_in_state)) {
            throw std::runtime_error("Dense gradient state expected to be vector");
        }
        const auto &grad_in = std::get<GradientVector>(grad_in_state);

        std::vector<std::vector<F>> weights_f(outputs, std::vector<F>(inputs, F_ZERO));
        for (std::size_t r = 0; r < outputs; ++r) {
            for (std::size_t c = 0; c < inputs; ++c) {
                weights_f[r][c] = quantize_scalar(layer.weights[r][c]);
            }
        }

        std::vector<F> grad_out_masked_f(outputs, F_ZERO);
        for (std::size_t i = 0; i < outputs; ++i) {
            grad_out_masked_f[i] = quantize_scalar(grad_out_masked[i]);
        }

        std::vector<F> grad_in_f(inputs, F_ZERO);
        for (std::size_t i = 0; i < inputs; ++i) {
            grad_in_f[i] = quantize_scalar(grad_in[i]);
        }

        // Transpose weights to match gradient propagation (W^T * grad_out_masked)
        std::vector<std::vector<F>> weights_t(inputs, std::vector<F>(outputs, F_ZERO));
        for (std::size_t r = 0; r < outputs; ++r) {
            for (std::size_t c = 0; c < inputs; ++c) {
                weights_t[c][r] = weights_f[r][c];
            }
        }

        const std::size_t rows_pad = next_power_of_two(inputs);
        const std::size_t cols_pad = next_power_of_two(outputs);

        auto weights_pad = pad_matrix(weights_t, rows_pad, cols_pad);
        auto grad_out_pad = pad_vector(grad_out_masked_f, cols_pad);
        auto vector_matrix = pad_matrix(vector_to_matrix(grad_out_pad), cols_pad, 1);
        auto result_pad = pad_vector(grad_in_f, rows_pad);

        proof pr = prove_matrix_vector_product(std::move(weights_pad), std::move(vector_matrix), result_pad);
        proofs.push_back(std::move(pr));
    }

    return proofs;
}

} // namespace

std::vector<LayerProof> prove_dense_layers(const Model &model,
                                           const GradientComputationResult &result) {
    if (!model.is_pure_dense()) {
        return {};
    }

    std::vector<LayerProof> proofs;
    proofs.reserve(model.layers.size());

    for (std::size_t idx = 0; idx < model.layers.size(); ++idx) {
        const auto &layer_variant = model.layers[idx];
        const auto *dense = std::get_if<DenseLayer>(&layer_variant);
        if (!dense) {
            throw std::runtime_error("Expected dense layer in prove_dense_layers");
        }

        const auto &cache_variant = result.forward_context.caches[idx];
        const auto &cache = std::get<DenseLayerCache>(cache_variant);

        std::vector<GradientState> next_states;
        if (idx + 1 < result.layer_sensitivities.size()) {
            next_states = result.layer_sensitivities[idx + 1].gradients_per_output;
        }

        LayerProof layer_proof;
        layer_proof.label = result.layer_sensitivities[idx].label;
        layer_proof.output_proofs = prove_dense_layer_outputs(*dense, cache, result.layer_sensitivities[idx], next_states);
        proofs.push_back(std::move(layer_proof));
    }

    return proofs;
}

} // namespace gradient
