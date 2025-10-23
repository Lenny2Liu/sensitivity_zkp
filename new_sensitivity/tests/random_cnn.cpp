#include "gradient/forward_backward.hpp"
#include "gradient/pipeline.hpp"
#include "gradient/zk_globals.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

using gradient::Conv2DLayer;
using gradient::Conv2DLayerConfig;
using gradient::DenseLayer;
using gradient::DenseLayerConfig;
using gradient::FlattenLayer;
using gradient::Model;
using gradient::TensorShape;

double random_double(std::mt19937 &rng, double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

std::vector<double> random_input(std::mt19937 &rng, std::size_t dim) {
    std::vector<double> input(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        input[i] = random_double(rng, -0.5, 0.5);
    }
    return input;
}

Model random_cnn_model(std::mt19937 &rng) {
    Model model;
    model.input_kind = Model::InputKind::Image;
    model.image_input_shape = TensorShape{1, 4, 4};

    Conv2DLayer conv;
    conv.config = Conv2DLayerConfig{1, 1, 3, 3, 1, 0, true};
    conv.weights.resize(conv.config.out_channels);
    for (auto &oc_weights : conv.weights) {
        oc_weights.resize(conv.config.in_channels);
        for (auto &ic_weights : oc_weights) {
            ic_weights.resize(conv.config.kernel_h);
            for (auto &row : ic_weights) {
                row.resize(conv.config.kernel_w);
                for (double &value : row) {
                    value = random_double(rng, -0.5, 0.5);
                }
            }
        }
    }
    conv.biases.resize(conv.config.out_channels);
    for (double &b : conv.biases) {
        b = random_double(rng, -0.25, 0.25);
    }

    model.layers.emplace_back(conv);
    model.layers.emplace_back(FlattenLayer{});

    const std::size_t out_height = model.image_input_shape.height - conv.config.kernel_h + 1;
    const std::size_t out_width = model.image_input_shape.width - conv.config.kernel_w + 1;
    const std::size_t flattened = conv.config.out_channels * out_height * out_width;

    DenseLayer dense;
    dense.config = DenseLayerConfig{flattened, 3, true};
    dense.weights.resize(dense.config.out_features, std::vector<double>(dense.config.in_features));
    dense.biases.resize(dense.config.out_features);
    for (std::size_t r = 0; r < dense.config.out_features; ++r) {
        for (std::size_t c = 0; c < dense.config.in_features; ++c) {
            dense.weights[r][c] = random_double(rng, -0.5, 0.5);
        }
        dense.biases[r] = random_double(rng, -0.25, 0.25);
    }
    model.layers.emplace_back(dense);

    return model;
}

std::vector<double> finite_difference_gradient(const Model &model,
                                               const std::vector<double> &input,
                                               std::size_t input_index) {
    const double eps = 1e-6;
    std::vector<double> plus = input;
    std::vector<double> minus = input;
    plus[input_index] += eps;
    minus[input_index] -= eps;

    auto plus_ctx = gradient::run_forward(model, plus);
    auto minus_ctx = gradient::run_forward(model, minus);

    std::vector<double> grad(plus_ctx.output.size());
    for (std::size_t j = 0; j < grad.size(); ++j) {
        grad[j] = (plus_ctx.output[j] - minus_ctx.output[j]) / (2.0 * eps);
    }
    return grad;
}

bool close(double a, double b, double tol = 1e-3) {
    return std::fabs(a - b) <= tol;
}

} // namespace

int main() {
    gradient::initialize_kaizen_globals();
    std::mt19937 rng(2024);
    bool ok = true;

    const int trials = 1;
    for (int trial = 0; trial < trials; ++trial) {
        auto model = random_cnn_model(rng);
        auto input = random_input(rng, model.flattened_input_size());
        std::uniform_int_distribution<std::size_t> idx_dist(0, input.size() - 1);
        std::size_t idx = idx_dist(rng);

        auto result = gradient::compute_gradient(model, input, idx);
        auto reference = finite_difference_gradient(model, input, idx);

        const std::size_t preview = std::min<std::size_t>(result.gradient_float.size(), 4);
        std::cout << "trial=" << trial << " input_idx=" << idx << " analytic:";
        for (std::size_t k = 0; k < preview; ++k) {
            std::cout << ' ' << std::fixed << std::setprecision(6) << result.gradient_float[k];
        }
        if (result.gradient_float.size() > preview) {
            std::cout << " ...";
        }
        std::cout << "\ntrial=" << trial << " input_idx=" << idx << " finite-diff:";
        for (std::size_t k = 0; k < preview; ++k) {
            std::cout << ' ' << std::fixed << std::setprecision(6) << reference[k];
        }
        if (reference.size() > preview) {
            std::cout << " ...";
        }
        std::cout << '\n';

        if (result.gradient_float.size() != reference.size()) {
            std::cerr << "Gradient size mismatch\n";
            return 1;
        }

        for (std::size_t j = 0; j < reference.size(); ++j) {
            if (!close(result.gradient_float[j], reference[j])) {
                std::cerr << "Mismatch at trial " << trial
                          << ", input idx " << idx
                          << ", output " << j
                          << ": pipeline=" << result.gradient_float[j]
                          << " reference=" << reference[j] << "\n";
                ok = false;
            }
        }
    }

    if (!ok) {
        return 1;
    }

    std::cout << "Random CNN gradient test passed\n";
    return 0;
}
