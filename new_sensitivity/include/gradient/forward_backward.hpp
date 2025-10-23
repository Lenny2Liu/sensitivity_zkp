#pragma once

#include "gradient/model.hpp"

#include <cstddef>
#include <variant>
#include <vector>

namespace gradient {

using ImageTensor = std::vector<std::vector<std::vector<double>>>; // [C][H][W]
using GradientVector = std::vector<double>;
using GradientTensor = ImageTensor;
using GradientState = std::variant<GradientVector, GradientTensor>;

struct DenseLayerCache {
    std::vector<double> input;
    std::vector<double> pre_activation;
    std::vector<double> mask;
};

struct ConvLayerCache {
    TensorShape input_shape;
    TensorShape output_shape;
    ImageTensor input;
    ImageTensor pre_activation;
    ImageTensor mask;
};

struct AvgPoolLayerCache {
    TensorShape input_shape;
    TensorShape output_shape;
    std::size_t kernel{};
    std::size_t stride{};
    ImageTensor input;
};

struct FlattenLayerCache {
    TensorShape original_shape;
    bool was_tensor{true};
};

using ForwardLayerCache = std::variant<DenseLayerCache, ConvLayerCache, AvgPoolLayerCache, FlattenLayerCache>;

struct ForwardContext {
    Model::InputKind input_kind;
    TensorShape input_shape;
    std::vector<double> input_flat;
    std::vector<double> output;
    std::vector<ForwardLayerCache> caches;
};

ForwardContext run_forward(const Model &model, const std::vector<double> &input_flat);

struct BackpropCoordinateResult {
    GradientState input_state;
    std::vector<GradientState> layer_states; // indexed by layer (0 -> first layer inputs)
};

BackpropCoordinateResult backprop_coordinate(const Model &model,
                                             const ForwardContext &context,
                                             std::size_t output_index);

double extract_input_gradient(Model::InputKind kind,
                              const TensorShape &input_shape,
                              const GradientState &state,
                              std::size_t input_index);

} // namespace gradient
