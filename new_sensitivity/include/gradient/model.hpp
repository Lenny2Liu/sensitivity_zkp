#pragma once

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace gradient {

struct TensorShape {
    std::size_t channels{};
    std::size_t height{};
    std::size_t width{};
};

struct DenseLayerConfig {
    std::size_t in_features{};
    std::size_t out_features{};
    bool apply_relu{true};
};

struct DenseLayer {
    DenseLayerConfig config;
    // weights[out][in]
    std::vector<std::vector<double>> weights;
    std::vector<double> biases; // length = out_features
};

struct Conv2DLayerConfig {
    std::size_t in_channels{};
    std::size_t out_channels{};
    std::size_t kernel_h{};
    std::size_t kernel_w{};
    std::size_t stride{1};
    std::size_t padding{0};
    bool apply_relu{true};
};

struct Conv2DLayer {
    Conv2DLayerConfig config;
    // weights[out_channel][in_channel][kernel_h][kernel_w]
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<double> biases; // length = out_channels
};

struct AvgPoolLayer {
    std::size_t kernel{};
    std::size_t stride{};
};

struct FlattenLayer {
};

using LayerVariant = std::variant<DenseLayer, Conv2DLayer, AvgPoolLayer, FlattenLayer>;

struct Model {
    enum class InputKind {
        Vector,
        Image
    };

    InputKind input_kind{InputKind::Vector};
    std::size_t vector_input_size{};
    TensorShape image_input_shape{};

    std::vector<LayerVariant> layers;

    [[nodiscard]] std::size_t input_dim() const;
    [[nodiscard]] std::size_t flattened_input_size() const;
    [[nodiscard]] std::size_t output_dim() const;
    [[nodiscard]] TensorShape input_shape() const;
    [[nodiscard]] bool is_pure_dense() const;
    [[nodiscard]] bool uses_convolutions() const;
};

Model load_from_file(const std::string &path);
void save_to_file(const Model &model, const std::string &path);

} // namespace gradient
