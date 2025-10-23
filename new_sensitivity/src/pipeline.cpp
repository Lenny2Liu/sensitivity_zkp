#include "gradient/pipeline.hpp"

#include "gradient/quantization.hpp"
#include "gradient/zk_prover.hpp"

#include <sstream>
#include <stdexcept>

namespace gradient {

namespace {

std::string layer_label(const LayerVariant &layer, std::size_t index) {
    std::ostringstream oss;
    oss << index << ": ";
    if (const auto *dense = std::get_if<DenseLayer>(&layer)) {
        oss << "dense(" << dense->config.in_features << "→" << dense->config.out_features
            << ", " << (dense->config.apply_relu ? "relu" : "linear") << ")";
    } else if (const auto *conv = std::get_if<Conv2DLayer>(&layer)) {
        oss << "conv2d(" << conv->config.in_channels << "→" << conv->config.out_channels
            << ", k=" << conv->config.kernel_h << "x" << conv->config.kernel_w
            << ", s=" << conv->config.stride
            << ", p=" << conv->config.padding
            << ", " << (conv->config.apply_relu ? "relu" : "linear") << ")";
    } else if (const auto *pool = std::get_if<AvgPoolLayer>(&layer)) {
        oss << "avgpool(k=" << pool->kernel << ", s=" << pool->stride << ")";
    } else {
        oss << "flatten";
    }
    return oss.str();
}

} // namespace

GradientComputationResult compute_gradient(const Model &model,
                                           const std::vector<double> &input,
                                           std::size_t input_index) {
    if (model.layers.empty()) {
        throw std::runtime_error("compute_gradient: model has no layers");
    }

    if (input.size() != model.flattened_input_size()) {
        throw std::runtime_error("compute_gradient: input dimension mismatch");
    }

    if (input_index >= model.flattened_input_size()) {
        throw std::runtime_error("compute_gradient: input index out of range");
    }

    ForwardContext ctx = run_forward(model, input);
    const std::size_t output_dim = ctx.output.size();

    GradientComputationResult result;
    result.layer_sensitivities.resize(model.layers.size());
    for (std::size_t layer_idx = 0; layer_idx < model.layers.size(); ++layer_idx) {
        auto &entry = result.layer_sensitivities[layer_idx];
        entry.label = layer_label(model.layers[layer_idx], layer_idx);
        entry.gradients_per_output.resize(output_dim);
    }

    result.gradient_float.resize(output_dim);
    for (std::size_t j = 0; j < output_dim; ++j) {
        BackpropCoordinateResult bp = backprop_coordinate(model, ctx, j);
        double grad_value = extract_input_gradient(model.input_kind, ctx.input_shape, bp.input_state, input_index);
        result.gradient_float[j] = grad_value;

        if (bp.layer_states.size() != model.layers.size()) {
            throw std::runtime_error("backprop state size mismatch with model layers");
        }
        for (std::size_t layer_idx = 0; layer_idx < model.layers.size(); ++layer_idx) {
            result.layer_sensitivities[layer_idx].gradients_per_output[j] = bp.layer_states[layer_idx];
        }
    }

    result.gradient.reserve(output_dim);
    for (double value : result.gradient_float) {
        result.gradient.push_back(quantize_scalar(value));
    }

    result.selected_input_index = input_index;
    result.input_float = input;
    result.quantized_input = quantize_vector(input);
    result.input_kind = model.input_kind;
    if (model.input_kind == Model::InputKind::Image) {
        result.input_shape = model.image_input_shape;
    }
    result.forward_context = ctx;
    if (model.is_pure_dense()) {
        result.layer_proofs = prove_dense_layers(model, result);
    }

    return result;
}

} // namespace gradient
