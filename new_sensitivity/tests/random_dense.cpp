#include "gradient/pipeline.hpp"
#include "gradient/zk_globals.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

using gradient::DenseLayer;
using gradient::DenseLayerConfig;
using gradient::Model;

std::vector<double> evaluate_dense(const Model &model, const std::vector<double> &input) {
    std::vector<double> state = input;
    for (const auto &variant : model.layers) {
        const auto &layer = std::get<DenseLayer>(variant);
        std::vector<double> next(layer.config.out_features, 0.0);
        for (std::size_t row = 0; row < layer.config.out_features; ++row) {
            double sum = layer.biases[row];
            for (std::size_t col = 0; col < layer.config.in_features; ++col) {
                sum += layer.weights[row][col] * state[col];
            }
            if (layer.config.apply_relu && sum <= 0.0) {
                sum = 0.0;
            }
            next[row] = sum;
        }
        state = std::move(next);
    }
    return state;
}

std::vector<double> finite_difference_gradient(const Model &model,
                                               const std::vector<double> &input,
                                               std::size_t input_index) {
    const double eps = 1e-6;
    std::vector<double> plus = input;
    std::vector<double> minus = input;
    plus[input_index] += eps;
    minus[input_index] -= eps;

    auto y_plus = evaluate_dense(model, plus);
    auto y_minus = evaluate_dense(model, minus);

    std::vector<double> grad(y_plus.size(), 0.0);
    for (std::size_t j = 0; j < grad.size(); ++j) {
        grad[j] = (y_plus[j] - y_minus[j]) / (2.0 * eps);
    }
    return grad;
}

double random_double(std::mt19937 &rng, double low, double high) {
    std::uniform_real_distribution<double> dist(low, high);
    return dist(rng);
}

Model random_model(std::mt19937 &rng) {
    std::uniform_int_distribution<int> layer_dist(1, 3);
    std::uniform_int_distribution<int> dim_dist(1, 4);
    std::bernoulli_distribution relu_dist(0.7);

    Model model;
    model.input_kind = Model::InputKind::Vector;
    int layers = layer_dist(rng);
    std::size_t in_dim = static_cast<std::size_t>(dim_dist(rng) + 1);
    model.vector_input_size = in_dim;
    for (int i = 0; i < layers; ++i) {
        std::size_t out_dim = static_cast<std::size_t>(dim_dist(rng) + 1);
        DenseLayer layer;
        layer.config = DenseLayerConfig{in_dim, out_dim, relu_dist(rng)};
        layer.weights.resize(out_dim, std::vector<double>(in_dim));
        layer.biases.resize(out_dim);
        for (std::size_t r = 0; r < out_dim; ++r) {
            for (std::size_t c = 0; c < in_dim; ++c) {
                layer.weights[r][c] = random_double(rng, -0.5, 0.5);
            }
            layer.biases[r] = random_double(rng, -0.25, 0.25);
        }
        model.layers.emplace_back(layer);
        in_dim = out_dim;
    }
    return model;
}

std::vector<double> random_input(std::mt19937 &rng, std::size_t dim) {
    std::vector<double> input(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        input[i] = random_double(rng, -1.0, 1.0);
        if (input[i] == 0.0) {
            input[i] = 1e-3;
        }
    }
    return input;
}

bool close(double a, double b, double tol = 1e-3) {
    return std::fabs(a - b) <= tol;
}

} // namespace

int main() {
    gradient::initialize_kaizen_globals();
    std::mt19937 rng(1337);
    bool ok = true;

    const int trials = 1;
    for (int trial = 0; trial < trials; ++trial) {
        auto model = random_model(rng);
        auto input = random_input(rng, model.input_dim());

        const std::size_t features_to_check = std::min<std::size_t>(input.size(), 2);
        for (std::size_t idx = 0; idx < features_to_check; ++idx) {
            auto pipeline_result = gradient::compute_gradient(model, input, idx);
            const auto &gradient_float = pipeline_result.gradient_float;
            auto reference = finite_difference_gradient(model, input, idx);

            if (pipeline_result.layer_proofs.empty()) {
                std::cerr << "Missing dense layer proofs" << std::endl;
                return 1;
            }

            const std::size_t preview = std::min<std::size_t>(gradient_float.size(), 4);
            std::cout << "trial=" << trial << " input_idx=" << idx << " analytic:";
            for (std::size_t k = 0; k < preview; ++k) {
                std::cout << ' ' << std::fixed << std::setprecision(6) << gradient_float[k];
            }
            if (gradient_float.size() > preview) {
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

            if (gradient_float.size() != reference.size()) {
                std::cerr << "Gradient size mismatch\n";
                return 1;
            }

            for (std::size_t j = 0; j < reference.size(); ++j) {
                if (!close(gradient_float[j], reference[j])) {
                    std::cerr << "Mismatch at trial " << trial << ", input idx " << idx
                              << ", output " << j << ": pipeline=" << gradient_float[j]
                              << " reference=" << reference[j] << '\n';
                    ok = false;
                }
            }
        }
    }

    if (!ok) {
        return 1;
    }

    std::cout << "Random dense gradient test passed\n";
    return 0;
}
