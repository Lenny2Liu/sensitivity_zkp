#include "gradient/forward_backward.hpp"

#include <cmath>
#include <stdexcept>

namespace gradient {
namespace {

using Vector = std::vector<double>;
using Tensor = ImageTensor;
using GradientState = std::variant<Vector, Tensor>;

Tensor make_tensor(const TensorShape &shape, double value = 0.0) {
    return Tensor(shape.channels,
                  std::vector<std::vector<double>>(shape.height,
                                                    std::vector<double>(shape.width, value)));
}

Vector flatten_tensor(const Tensor &tensor) {
    Vector flat;
    const std::size_t channels = tensor.size();
    if (channels == 0) {
        return flat;
    }
    const std::size_t height = tensor[0].size();
    const std::size_t width = height > 0 ? tensor[0][0].size() : 0;
    flat.reserve(channels * height * width);
    for (const auto &channel : tensor) {
        for (const auto &row : channel) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    return flat;
}

Tensor reshape_tensor(const Vector &flat, const TensorShape &shape) {
    const std::size_t expected = shape.channels * shape.height * shape.width;
    if (flat.size() != expected) {
        throw std::runtime_error("reshape_tensor: size mismatch");
    }
    Tensor tensor = make_tensor(shape, 0.0);
    std::size_t idx = 0;
    for (std::size_t c = 0; c < shape.channels; ++c) {
        for (std::size_t h = 0; h < shape.height; ++h) {
            for (std::size_t w = 0; w < shape.width; ++w) {
                tensor[c][h][w] = flat[idx++];
            }
        }
    }
    return tensor;
}

TensorShape conv_output_shape(const TensorShape &input, const Conv2DLayerConfig &cfg) {
    if (cfg.stride == 0) {
        throw std::runtime_error("conv stride must be positive");
    }
    if (cfg.kernel_h == 0 || cfg.kernel_w == 0) {
        throw std::runtime_error("conv kernel size must be positive");
    }
    if (cfg.in_channels != input.channels) {
        throw std::runtime_error("conv layer channel mismatch");
    }
    const std::size_t padded_h = input.height + 2 * cfg.padding;
    const std::size_t padded_w = input.width + 2 * cfg.padding;
    if (padded_h < cfg.kernel_h || padded_w < cfg.kernel_w) {
        throw std::runtime_error("conv kernel larger than padded input");
    }
    const std::size_t out_h_num = padded_h - cfg.kernel_h;
    const std::size_t out_w_num = padded_w - cfg.kernel_w;
    if (out_h_num % cfg.stride != 0 || out_w_num % cfg.stride != 0) {
        throw std::runtime_error("conv stride does not align with input dimensions");
    }
    return TensorShape{
        cfg.out_channels,
        out_h_num / cfg.stride + 1,
        out_w_num / cfg.stride + 1
    };
}

TensorShape avgpool_output_shape(const TensorShape &input, const AvgPoolLayer &layer) {
    if (layer.kernel == 0 || layer.stride == 0) {
        throw std::runtime_error("avgpool kernel/stride must be positive");
    }
    if (input.height < layer.kernel || input.width < layer.kernel) {
        throw std::runtime_error("avgpool kernel larger than input");
    }
    const std::size_t out_h_num = input.height - layer.kernel;
    const std::size_t out_w_num = input.width - layer.kernel;
    if (out_h_num % layer.stride != 0 || out_w_num % layer.stride != 0) {
        throw std::runtime_error("avgpool stride does not align with input dimensions");
    }
    return TensorShape{
        input.channels,
        out_h_num / layer.stride + 1,
        out_w_num / layer.stride + 1
    };
}

void apply_relu(Vector &values, Vector &mask) {
    mask.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] <= 0.0) {
            values[i] = 0.0;
            mask[i] = 0.0;
        } else {
            mask[i] = 1.0;
        }
    }
}

void apply_relu(Tensor &values, Tensor &mask) {
    mask = make_tensor(TensorShape{values.size(),
                                   values.empty() ? 0 : values[0].size(),
                                   values.empty() || values[0].empty() ? 0 : values[0][0].size()},
                       0.0);
    for (std::size_t c = 0; c < values.size(); ++c) {
        for (std::size_t h = 0; h < values[c].size(); ++h) {
            for (std::size_t w = 0; w < values[c][h].size(); ++w) {
                if (values[c][h][w] <= 0.0) {
                    values[c][h][w] = 0.0;
                    mask[c][h][w] = 0.0;
                } else {
                    mask[c][h][w] = 1.0;
                }
            }
        }
    }
}

Vector dense_forward(const Vector &input, const DenseLayer &layer, Vector &pre_activation) {
    Vector output(layer.config.out_features, 0.0);
    pre_activation.resize(layer.config.out_features);
    for (std::size_t row = 0; row < layer.config.out_features; ++row) {
        double sum = layer.biases[row];
        const auto &weights = layer.weights[row];
        for (std::size_t col = 0; col < layer.config.in_features; ++col) {
            sum += weights[col] * input[col];
        }
        pre_activation[row] = sum;
        output[row] = sum;
    }
    return output;
}

Vector ensure_vector(const GradientState &state) {
    if (!std::holds_alternative<Vector>(state)) {
        throw std::runtime_error("gradient state expected to be vector");
    }
    return std::get<Vector>(state);
}

Tensor ensure_tensor(const GradientState &state) {
    if (!std::holds_alternative<Tensor>(state)) {
        throw std::runtime_error("gradient state expected to be tensor");
    }
    return std::get<Tensor>(state);
}

void index_to_coords(const TensorShape &shape,
                     std::size_t index,
                     std::size_t &channel,
                     std::size_t &row,
                     std::size_t &col) {
    const std::size_t per_channel = shape.height * shape.width;
    channel = index / per_channel;
    const std::size_t remainder = index % per_channel;
    row = remainder / shape.width;
    col = remainder % shape.width;
    if (channel >= shape.channels || row >= shape.height || col >= shape.width) {
        throw std::runtime_error("input index out of bounds for tensor shape");
    }
}

} // namespace

ForwardContext run_forward(const Model &model, const std::vector<double> &input_flat) {
    if (model.input_kind == Model::InputKind::Vector) {
        if (input_flat.size() != model.vector_input_size) {
            throw std::runtime_error("run_forward: vector input size mismatch");
        }
    } else if (input_flat.size() != model.flattened_input_size()) {
        throw std::runtime_error("run_forward: image input size mismatch");
    }

    ForwardContext ctx;
    ctx.input_kind = model.input_kind;
    ctx.input_shape = model.image_input_shape;
    ctx.input_flat = input_flat;
    ctx.caches.reserve(model.layers.size());

    struct ActivationState {
        bool is_vector;
        Vector vector;
        Tensor tensor;
        TensorShape shape;
    };

    ActivationState state;
    if (model.input_kind == Model::InputKind::Vector) {
        state.is_vector = true;
        state.vector = input_flat;
    } else {
        state.is_vector = false;
        state.shape = model.image_input_shape;
        state.tensor = reshape_tensor(input_flat, state.shape);
    }

    for (const auto &layer_variant : model.layers) {
        if (const auto *dense = std::get_if<DenseLayer>(&layer_variant)) {
            if (!state.is_vector) {
                throw std::runtime_error("dense layer received tensor input without flatten");
            }
            DenseLayerCache cache;
            cache.input = state.vector;

            Vector pre;
            Vector output = dense_forward(state.vector, *dense, pre);
            cache.pre_activation = std::move(pre);
            cache.mask.assign(output.size(), 1.0);
            if (dense->config.apply_relu) {
                apply_relu(output, cache.mask);
            }

            ctx.caches.emplace_back(std::move(cache));
            state.vector = std::move(output);
            state.is_vector = true;
        } else if (const auto *conv = std::get_if<Conv2DLayer>(&layer_variant)) {
            if (state.is_vector) {
                throw std::runtime_error("conv2d layer expects image input");
            }
            ConvLayerCache cache;
            cache.input_shape = state.shape;
            cache.input = state.tensor;

            Tensor pre;
            Tensor output = make_tensor(conv_output_shape(state.shape, conv->config), 0.0);
            pre = make_tensor(conv_output_shape(state.shape, conv->config), 0.0);

            for (std::size_t oc = 0; oc < conv->config.out_channels; ++oc) {
                for (std::size_t oh = 0; oh < output.size() ? output[oc].size() : 0; ++oh) {
                    for (std::size_t ow = 0; ow < output[oc][oh].size(); ++ow) {
                        double sum = conv->biases[oc];
                        for (std::size_t ic = 0; ic < conv->config.in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < conv->config.kernel_h; ++kh) {
                                for (std::size_t kw = 0; kw < conv->config.kernel_w; ++kw) {
                                    const int ih = static_cast<int>(oh * conv->config.stride + kh) - static_cast<int>(conv->config.padding);
                                    const int iw = static_cast<int>(ow * conv->config.stride + kw) - static_cast<int>(conv->config.padding);
                                    if (ih < 0 || iw < 0 ||
                                        ih >= static_cast<int>(state.shape.height) ||
                                        iw >= static_cast<int>(state.shape.width)) {
                                        continue;
                                    }
                                    sum += conv->weights[oc][ic][kh][kw] * state.tensor[ic][ih][iw];
                                }
                            }
                        }
                        pre[oc][oh][ow] = sum;
                        output[oc][oh][ow] = sum;
                    }
                }
            }

            cache.pre_activation = pre;
            cache.output_shape = TensorShape{conv->config.out_channels,
                                             pre.empty() ? 0 : pre[0].size(),
                                             pre.empty() || pre[0].empty() ? 0 : pre[0][0].size()};
            cache.mask = make_tensor(cache.output_shape, 1.0);
            if (conv->config.apply_relu) {
                apply_relu(output, cache.mask);
            }

            ctx.caches.emplace_back(std::move(cache));
            state.tensor = std::move(output);
            state.shape = TensorShape{conv->config.out_channels,
                                      cache.output_shape.height,
                                      cache.output_shape.width};
            state.is_vector = false;
        } else if (const auto *pool = std::get_if<AvgPoolLayer>(&layer_variant)) {
            if (state.is_vector) {
                throw std::runtime_error("avgpool layer expects image input");
            }
            AvgPoolLayerCache cache;
            cache.input_shape = state.shape;
            cache.input = state.tensor;
            cache.kernel = pool->kernel;
            cache.stride = pool->stride;
            cache.output_shape = avgpool_output_shape(state.shape, *pool);

            Tensor output = make_tensor(cache.output_shape, 0.0);
            for (std::size_t c = 0; c < state.shape.channels; ++c) {
                for (std::size_t oh = 0; oh < cache.output_shape.height; ++oh) {
                    for (std::size_t ow = 0; ow < cache.output_shape.width; ++ow) {
                        double sum = 0.0;
                        for (std::size_t kh = 0; kh < pool->kernel; ++kh) {
                            for (std::size_t kw = 0; kw < pool->kernel; ++kw) {
                                const std::size_t ih = oh * pool->stride + kh;
                                const std::size_t iw = ow * pool->stride + kw;
                                sum += state.tensor[c][ih][iw];
                            }
                        }
                        output[c][oh][ow] = sum / static_cast<double>(pool->kernel * pool->kernel);
                    }
                }
            }

            ctx.caches.emplace_back(std::move(cache));
            state.tensor = std::move(output);
            state.shape = cache.output_shape;
            state.is_vector = false;
        } else if (std::holds_alternative<FlattenLayer>(layer_variant)) {
            FlattenLayerCache cache;
            cache.was_tensor = !state.is_vector;
            if (!state.is_vector) {
                cache.original_shape = state.shape;
                Vector flat = flatten_tensor(state.tensor);
                state.vector = std::move(flat);
                state.is_vector = true;
            } else {
                cache.original_shape = TensorShape{1, 1, state.vector.size()};
            }
            ctx.caches.emplace_back(std::move(cache));
        }
    }

    if (!state.is_vector) {
        throw std::runtime_error("model output is not a vector; missing flatten/dense at end");
    }
    ctx.output = state.vector;
    return ctx;
}

BackpropCoordinateResult backprop_coordinate(const Model &model,
                                             const ForwardContext &context,
                                             std::size_t output_index) {
    if (output_index >= context.output.size()) {
        throw std::runtime_error("output index out of range");
    }
    GradientState grad_state = Vector(context.output.size(), 0.0);
    std::get<Vector>(grad_state)[output_index] = 1.0;

    if (model.layers.size() != context.caches.size()) {
        throw std::runtime_error("forward context mismatch with model layers");
    }

    std::vector<GradientState> layer_states(model.layers.size());

    for (std::size_t rev = 0; rev < model.layers.size(); ++rev) {
        const std::size_t layer_idx = model.layers.size() - 1 - rev;
        const auto &layer_variant = model.layers[layer_idx];
        const auto &cache_variant = context.caches[layer_idx];

        if (const auto *dense = std::get_if<DenseLayer>(&layer_variant)) {
            Vector grad = ensure_vector(grad_state);
            const auto &cache = std::get<DenseLayerCache>(cache_variant);
            if (dense->config.apply_relu) {
                for (std::size_t i = 0; i < grad.size(); ++i) {
                    grad[i] *= cache.mask[i];
                }
            }
            Vector prev(cache.input.size(), 0.0);
            for (std::size_t row = 0; row < dense->config.out_features; ++row) {
                const double g = grad[row];
                const auto &weights = dense->weights[row];
                for (std::size_t col = 0; col < dense->config.in_features; ++col) {
                    prev[col] += weights[col] * g;
                }
            }
            layer_states[layer_idx] = prev;
            grad_state = std::move(prev);
        } else if (const auto *conv = std::get_if<Conv2DLayer>(&layer_variant)) {
            Tensor grad = ensure_tensor(grad_state);
            const auto &cache = std::get<ConvLayerCache>(cache_variant);
            if (conv->config.apply_relu) {
                for (std::size_t c = 0; c < grad.size(); ++c) {
                    for (std::size_t h = 0; h < grad[c].size(); ++h) {
                        for (std::size_t w = 0; w < grad[c][h].size(); ++w) {
                            grad[c][h][w] *= cache.mask[c][h][w];
                        }
                    }
                }
            }
            Tensor prev = make_tensor(cache.input_shape, 0.0);
            for (std::size_t oc = 0; oc < conv->config.out_channels; ++oc) {
                for (std::size_t oh = 0; oh < grad[oc].size(); ++oh) {
                    for (std::size_t ow = 0; ow < grad[oc][oh].size(); ++ow) {
                        const double g = grad[oc][oh][ow];
                        for (std::size_t ic = 0; ic < conv->config.in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < conv->config.kernel_h; ++kh) {
                                for (std::size_t kw = 0; kw < conv->config.kernel_w; ++kw) {
                                    const int ih = static_cast<int>(oh * conv->config.stride + kh) - static_cast<int>(conv->config.padding);
                                    const int iw = static_cast<int>(ow * conv->config.stride + kw) - static_cast<int>(conv->config.padding);
                                    if (ih < 0 || iw < 0 ||
                                        ih >= static_cast<int>(cache.input_shape.height) ||
                                        iw >= static_cast<int>(cache.input_shape.width)) {
                                        continue;
                                    }
                                    prev[ic][ih][iw] += conv->weights[oc][ic][kh][kw] * g;
                                }
                            }
                        }
                    }
                }
            }
            layer_states[layer_idx] = prev;
            grad_state = std::move(prev);
        } else if (const auto *pool = std::get_if<AvgPoolLayer>(&layer_variant)) {
            Tensor grad = ensure_tensor(grad_state);
            const auto &cache = std::get<AvgPoolLayerCache>(cache_variant);
            Tensor prev = make_tensor(cache.input_shape, 0.0);

            for (std::size_t c = 0; c < cache.output_shape.channels; ++c) {
                for (std::size_t oh = 0; oh < cache.output_shape.height; ++oh) {
                    for (std::size_t ow = 0; ow < cache.output_shape.width; ++ow) {
                        const double g = grad[c][oh][ow] / static_cast<double>(pool->kernel * pool->kernel);
                        for (std::size_t kh = 0; kh < pool->kernel; ++kh) {
                            for (std::size_t kw = 0; kw < pool->kernel; ++kw) {
                                const std::size_t ih = oh * pool->stride + kh;
                                const std::size_t iw = ow * pool->stride + kw;
                                prev[c][ih][iw] += g;
                            }
                        }
                    }
                }
            }
            layer_states[layer_idx] = prev;
            grad_state = std::move(prev);
        } else if (std::holds_alternative<FlattenLayer>(layer_variant)) {
            const auto &cache = std::get<FlattenLayerCache>(cache_variant);
            if (cache.was_tensor) {
                Vector grad = ensure_vector(grad_state);
                GradientState prev = reshape_tensor(grad, cache.original_shape);
                layer_states[layer_idx] = prev;
                grad_state = std::move(prev);
            } else {
                // Flatten applied to vector; no transformation needed.
                layer_states[layer_idx] = ensure_vector(grad_state);
            }
        }
    }

    BackpropCoordinateResult result;
    result.input_state = grad_state;
    result.layer_states = std::move(layer_states);
    return result;
}

double extract_input_gradient(Model::InputKind kind,
                              const TensorShape &input_shape,
                              const GradientState &state,
                              std::size_t input_index) {
    if (kind == Model::InputKind::Vector) {
        const auto &grad = std::get<GradientVector>(state);
        if (input_index >= grad.size()) {
            throw std::runtime_error("input index out of range");
        }
        return grad[input_index];
    }

    const auto &grad = std::get<GradientTensor>(state);
    std::size_t channel = 0;
    std::size_t row = 0;
    std::size_t col = 0;
    index_to_coords(input_shape, input_index, channel, row, col);
    return grad[channel][row][col];
}

} // namespace gradient
