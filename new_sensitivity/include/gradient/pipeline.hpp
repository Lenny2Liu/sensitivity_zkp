#pragma once

#include "GKR.h"
#include "config_pc.hpp"
#include "gradient/forward_backward.hpp"
#include "gradient/model.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gradient {

struct LayerSensitivity {
    std::string label;
    std::vector<GradientState> gradients_per_output; // size = number of outputs
};

struct LayerProof {
    std::string label;
    std::vector<proof> output_proofs;
};

struct GradientComputationResult {
    std::vector<F> gradient;                 // dy_j/dx_i in field representation
    std::vector<double> gradient_float;      // dy_j/dx_i as doubles
    std::size_t selected_input_index{};
    std::vector<F> quantized_input;
    std::vector<double> input_float;
    Model::InputKind input_kind;
    std::optional<TensorShape> input_shape;
    ForwardContext forward_context;
    std::vector<LayerSensitivity> layer_sensitivities;
    std::vector<LayerProof> layer_proofs;
};

GradientComputationResult compute_gradient(const Model &model,
                                           const std::vector<double> &input,
                                           std::size_t input_index);

} // namespace gradient
