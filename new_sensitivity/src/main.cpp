#include "gradient/gkr_plan.hpp"
#include "gradient/model.hpp"
#include "gradient/pipeline.hpp"
#include "gradient/quantization.hpp"
#include "gradient/zk_globals.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using gradient::Model;
using gradient::compute_gradient;

namespace {

std::vector<double> flatten_state(const gradient::GradientState &state) {
    if (std::holds_alternative<gradient::GradientVector>(state)) {
        return std::get<gradient::GradientVector>(state);
    }
    const auto &tensor = std::get<gradient::GradientTensor>(state);
    std::vector<double> values;
    for (const auto &channel : tensor) {
        for (const auto &row : channel) {
            values.insert(values.end(), row.begin(), row.end());
        }
    }
    return values;
}

std::string state_shape_desc(const gradient::GradientState &state) {
    if (std::holds_alternative<gradient::GradientVector>(state)) {
        return "vector(len=" + std::to_string(std::get<gradient::GradientVector>(state).size()) + ")";
    }
    const auto &tensor = std::get<gradient::GradientTensor>(state);
    const std::size_t channels = tensor.size();
    const std::size_t height = channels ? tensor[0].size() : 0;
    const std::size_t width = (channels && height) ? tensor[0][0].size() : 0;
    std::ostringstream oss;
    oss << "tensor(C=" << channels << ", H=" << height << ", W=" << width << ")";
    return oss.str();
}

void print_values_line(const std::vector<double> &values, std::size_t indent) {
    const std::string padding(indent, ' ');
    std::cout << padding << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << std::setprecision(6) << std::fixed << values[i];
        if (i + 1 != values.size()) {
            std::cout << ',';
        }
    }
    std::cout << "]" << std::endl;
}

void print_gradient_state(const gradient::GradientState &state, std::size_t indent) {
    auto values = flatten_state(state);
    double l1 = 0.0;
    double l2_sq = 0.0;
    double linf = 0.0;
    for (double v : values) {
        const double abs_v = std::fabs(v);
        l1 += abs_v;
        l2_sq += v * v;
        if (abs_v > linf) {
            linf = abs_v;
        }
    }
    const double l2 = std::sqrt(l2_sq);

    const std::string padding(indent, ' ');
    std::cout << padding << state_shape_desc(state) << " | L1=" << l1
              << ", L2=" << l2 << ", Linf=" << linf << std::endl;

    if (values.empty()) {
        std::cout << padding << "[empty]" << std::endl;
        return;
    }

    // Print values with 8 entries per line for readability
    const std::size_t per_line = 8;
    for (std::size_t i = 0; i < values.size(); i += per_line) {
        std::size_t end = std::min(values.size(), i + per_line);
        std::vector<double> slice(values.begin() + static_cast<long>(i), values.begin() + static_cast<long>(end));
        print_values_line(slice, indent + 2);
    }
}

} // namespace

std::string int128_to_string(__int128 value) {
    if (value == 0) {
        return "0";
    }
    bool negative = value < 0;
    __int128 tmp = negative ? -value : value;
    std::string digits;
    while (tmp > 0) {
        int digit = static_cast<int>(tmp % 10);
        digits.push_back(static_cast<char>('0' + digit));
        tmp /= 10;
    }
    if (negative) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

std::vector<double> load_input_vector(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("unable to open input file: " + path);
    }
    std::vector<double> values;
    double v;
    while (in >> v) {
        values.push_back(v);
    }
    if (values.empty()) {
        throw std::runtime_error("input vector is empty");
    }
    return values;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_file> <input_vector_file> <input_index>" << std::endl;
        return 1;
    }

    try {
        gradient::initialize_kaizen_globals();
        const std::string model_path = argv[1];
        const std::string input_path = argv[2];
        const std::size_t input_index = static_cast<std::size_t>(std::stoul(argv[3]));

        Model model = gradient::load_from_file(model_path);
        auto input = load_input_vector(input_path);
        if (input.size() != model.flattened_input_size()) {
            throw std::runtime_error("input vector length does not match model input dimension");
        }
        if (input_index >= input.size()) {
            throw std::runtime_error("input index out of range");
        }

        auto result = compute_gradient(model, input, input_index);

        std::cout << "Gradient dy/dx_" << input_index << " (field representation):" << std::endl;
        for (std::size_t j = 0; j < result.gradient.size(); ++j) {
            std::cout << "  y_" << j << ": " << int128_to_string(result.gradient[j].toint128()) << std::endl;
        }

        std::cout << "Gradient (approximate float):" << std::endl;
        for (std::size_t j = 0; j < result.gradient_float.size(); ++j) {
            std::cout << "  y_" << j << ": " << result.gradient_float[j] << std::endl;
        }

        double l1 = 0.0;
        double l2_sq = 0.0;
        double linf = 0.0;
        for (double v : result.gradient_float) {
            const double abs_v = std::fabs(v);
            l1 += abs_v;
            l2_sq += v * v;
            if (abs_v > linf) {
                linf = abs_v;
            }
        }
        const double l2 = std::sqrt(l2_sq);

        std::cout << "Gradient statistics:" << std::endl;
        std::cout << "  L1 norm: " << l1 << std::endl;
        std::cout << "  L2 norm: " << l2 << std::endl;
        std::cout << "  Linf norm: " << linf << std::endl;

        if (model.input_kind == Model::InputKind::Image) {
            const auto shape = model.input_shape();
            const std::size_t per_channel = shape.height * shape.width;
            std::size_t channel = input_index / per_channel;
            const std::size_t rem = input_index % per_channel;
            const std::size_t row = rem / shape.width;
            const std::size_t col = rem % shape.width;
            std::cout << "Input feature coordinate: channel=" << channel
                      << ", row=" << row
                      << ", col=" << col << std::endl;
        }

        if (!result.layer_sensitivities.empty()) {
            std::cout << "Per-layer sensitivities (gradients of each output w.r.t. layer inputs):" << std::endl;
            for (const auto &layer : result.layer_sensitivities) {
                std::cout << "Layer " << layer.label << std::endl;
                for (std::size_t out_idx = 0; out_idx < layer.gradients_per_output.size(); ++out_idx) {
                    std::cout << "  Output y_" << out_idx << ":" << std::endl;
                    print_gradient_state(layer.gradients_per_output[out_idx], 4);
                }
            }
        }

        const auto gkr_plan = gradient::build_gkr_plan(model, result);
        if (!gkr_plan.empty()) {
            std::cout << "GKR wiring plan:" << std::endl;
            for (const auto &item : gkr_plan) {
                std::cout << "  - " << item.layer_label << " -> " << item.description << std::endl;
            }
        }

        if (!result.layer_proofs.empty()) {
            std::cout << "Generated dense layer proofs:" << std::endl;
            for (const auto &entry : result.layer_proofs) {
                std::cout << "  Layer " << entry.label << ": "
                          << entry.output_proofs.size() << " proof(s)" << std::endl;
            }
        }

        // TODO: invoke GKR-based prover to produce a zero-knowledge transcript for the backward computation.
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
