#include "gradient/model.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using gradient::AvgPoolLayer;
using gradient::Conv2DLayer;
using gradient::DenseLayer;
using gradient::FlattenLayer;
using gradient::LayerVariant;
using gradient::Model;
using gradient::TensorShape;

TensorShape conv_output_shape(const TensorShape &input, const gradient::Conv2DLayerConfig &cfg) {
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

std::string trim(const std::string &str) {
    const auto first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

void ensure_stream_ok(std::istream &in, const std::string &message) {
    if (!in) {
        throw std::runtime_error(message);
    }
}

DenseLayer parse_dense_layer(std::istream &in) {
    std::size_t in_features = 0;
    std::size_t out_features = 0;
    std::string activation;
    if (!(in >> in_features >> out_features >> activation)) {
        throw std::runtime_error("failed to read dense layer header");
    }

    DenseLayer layer;
    layer.config.in_features = in_features;
    layer.config.out_features = out_features;
    layer.config.apply_relu = trim(activation) != "linear";

    layer.weights.resize(out_features, std::vector<double>(in_features));
    for (std::size_t row = 0; row < out_features; ++row) {
        for (std::size_t col = 0; col < in_features; ++col) {
            if (!(in >> layer.weights[row][col])) {
                throw std::runtime_error("failed to read dense weight value");
            }
        }
    }

    layer.biases.resize(out_features);
    for (std::size_t i = 0; i < out_features; ++i) {
        if (!(in >> layer.biases[i])) {
            throw std::runtime_error("failed to read dense bias value");
        }
    }

    return layer;
}

Conv2DLayer parse_conv_layer(std::istream &in) {
    Conv2DLayer layer;
    auto &cfg = layer.config;
    std::string activation;
    if (!(in >> cfg.in_channels >> cfg.out_channels >> cfg.kernel_h >> cfg.kernel_w >>
          cfg.stride >> cfg.padding >> activation)) {
        throw std::runtime_error("failed to read conv2d layer header");
    }
    cfg.apply_relu = trim(activation) != "linear";

    layer.weights.resize(cfg.out_channels);
    for (std::size_t oc = 0; oc < cfg.out_channels; ++oc) {
        layer.weights[oc].resize(cfg.in_channels);
        for (std::size_t ic = 0; ic < cfg.in_channels; ++ic) {
            layer.weights[oc][ic].resize(cfg.kernel_h);
            for (std::size_t kh = 0; kh < cfg.kernel_h; ++kh) {
                layer.weights[oc][ic][kh].resize(cfg.kernel_w);
                for (std::size_t kw = 0; kw < cfg.kernel_w; ++kw) {
                    if (!(in >> layer.weights[oc][ic][kh][kw])) {
                        throw std::runtime_error("failed to read conv weight value");
                    }
                }
            }
        }
    }

    layer.biases.resize(cfg.out_channels);
    for (std::size_t oc = 0; oc < cfg.out_channels; ++oc) {
        if (!(in >> layer.biases[oc])) {
            throw std::runtime_error("failed to read conv bias value");
        }
    }

    return layer;
}

AvgPoolLayer parse_avgpool_layer(std::istream &in) {
    AvgPoolLayer layer{};
    if (!(in >> layer.kernel >> layer.stride)) {
        throw std::runtime_error("failed to read avgpool layer");
    }
    return layer;
}

} // namespace

namespace gradient {

std::size_t Model::flattened_input_size() const {
    if (input_kind == InputKind::Vector) {
        return vector_input_size;
    }
    return image_input_shape.channels * image_input_shape.height * image_input_shape.width;
}

std::size_t Model::input_dim() const {
    return flattened_input_size();
}

TensorShape Model::input_shape() const {
    if (input_kind != InputKind::Image) {
        throw std::runtime_error("model input is not an image");
    }
    return image_input_shape;
}

bool Model::is_pure_dense() const {
    if (input_kind != InputKind::Vector) {
        return false;
    }
    for (const auto &layer : layers) {
        if (!std::holds_alternative<DenseLayer>(layer)) {
            return false;
        }
    }
    return true;
}

bool Model::uses_convolutions() const {
    for (const auto &layer : layers) {
        if (std::holds_alternative<Conv2DLayer>(layer) || std::holds_alternative<AvgPoolLayer>(layer)) {
            return true;
        }
    }
    return false;
}

std::size_t Model::output_dim() const {
    if (layers.empty()) {
        throw std::runtime_error("model has no layers");
    }
    bool is_vector = (input_kind == InputKind::Vector);
    std::size_t vector_dim = vector_input_size;
    TensorShape tensor_shape = image_input_shape;

    for (const auto &layer : layers) {
        if (const auto *dense = std::get_if<DenseLayer>(&layer)) {
            if (!is_vector) {
                throw std::runtime_error("dense layer expects vector input (missing flatten)");
            }
            vector_dim = dense->config.out_features;
            is_vector = true;
        } else if (const auto *conv = std::get_if<Conv2DLayer>(&layer)) {
            if (is_vector) {
                throw std::runtime_error("conv2d layer expects image input");
            }
            tensor_shape = conv_output_shape(tensor_shape, conv->config);
            is_vector = false;
        } else if (const auto *pool = std::get_if<AvgPoolLayer>(&layer)) {
            if (is_vector) {
                throw std::runtime_error("avgpool layer expects image input");
            }
            tensor_shape = avgpool_output_shape(tensor_shape, *pool);
            is_vector = false;
        } else if (std::holds_alternative<FlattenLayer>(layer)) {
            if (!is_vector) {
                vector_dim = tensor_shape.channels * tensor_shape.height * tensor_shape.width;
                is_vector = true;
            }
        }
    }

    if (!is_vector) {
        throw std::runtime_error("model output is not a vector; add a flatten or dense layer at the end");
    }
    return vector_dim;
}

Model load_from_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("unable to open model file: " + path);
    }

    Model model;
    std::string first_token;
    if (!(in >> first_token)) {
        throw std::runtime_error("empty model file");
    }

    if (first_token == "input") {
        std::string input_type;
        if (!(in >> input_type)) {
            throw std::runtime_error("expected input type after 'input'");
        }
        if (input_type == "vector") {
            if (!(in >> model.vector_input_size)) {
                throw std::runtime_error("failed to read vector input size");
            }
            model.input_kind = Model::InputKind::Vector;
        } else if (input_type == "image") {
            if (!(in >> model.image_input_shape.channels >> model.image_input_shape.height >> model.image_input_shape.width)) {
                throw std::runtime_error("failed to read image input shape");
            }
            model.input_kind = Model::InputKind::Image;
        } else {
            throw std::runtime_error("unknown input type: " + input_type);
        }

        std::string token;
        while (in >> token) {
            if (token != "layer") {
                throw std::runtime_error("expected 'layer' token but found '" + token + "'");
            }
            std::string layer_type;
            if (!(in >> layer_type)) {
                throw std::runtime_error("failed to read layer type");
            }
            if (layer_type == "dense") {
                DenseLayer dense = parse_dense_layer(in);
                model.layers.emplace_back(std::move(dense));
            } else if (layer_type == "conv2d") {
                Conv2DLayer conv = parse_conv_layer(in);
                model.layers.emplace_back(std::move(conv));
            } else if (layer_type == "avgpool") {
                AvgPoolLayer pool = parse_avgpool_layer(in);
                model.layers.emplace_back(pool);
            } else if (layer_type == "flatten") {
                model.layers.emplace_back(FlattenLayer{});
            } else {
                throw std::runtime_error("unknown layer type: " + layer_type);
            }
        }

        if (model.layers.empty()) {
            throw std::runtime_error("model contains no layers");
        }
        if (model.input_kind == Model::InputKind::Vector && model.vector_input_size == 0) {
            const auto *dense = std::get_if<DenseLayer>(&model.layers.front());
            if (!dense) {
                throw std::runtime_error("vector model must start with a dense layer");
            }
            model.vector_input_size = dense->config.in_features;
        }
        return model;
    }

    // Legacy dense-only format: first token is the number of layers.
    std::size_t layer_count = 0;
    try {
        layer_count = static_cast<std::size_t>(std::stoul(first_token));
    } catch (const std::exception &) {
        throw std::runtime_error("invalid legacy model header: expected layer count");
    }

    model.input_kind = Model::InputKind::Vector;
    for (std::size_t idx = 0; idx < layer_count; ++idx) {
        DenseLayer layer = parse_dense_layer(in);
        if (idx == 0) {
            model.vector_input_size = layer.config.in_features;
        }
        model.layers.emplace_back(std::move(layer));
    }

    if (model.layers.empty()) {
        throw std::runtime_error("legacy model contains no layers");
    }
    return model;
}

namespace {

struct LayerWriter {
    std::ostream &out;

    void operator()(const DenseLayer &layer) const {
        out << "layer dense "
            << layer.config.in_features << ' '
            << layer.config.out_features << ' '
            << (layer.config.apply_relu ? "relu" : "linear") << "\n";
        for (const auto &row : layer.weights) {
            for (std::size_t col = 0; col < row.size(); ++col) {
                out << row[col];
                if (col + 1 != row.size()) {
                    out << ' ';
                }
            }
            out << "\n";
        }
        for (std::size_t i = 0; i < layer.biases.size(); ++i) {
            out << layer.biases[i];
            if (i + 1 != layer.biases.size()) {
                out << ' ';
            }
        }
        out << "\n";
    }

    void operator()(const Conv2DLayer &layer) const {
        const auto &cfg = layer.config;
        out << "layer conv2d "
            << cfg.in_channels << ' '
            << cfg.out_channels << ' '
            << cfg.kernel_h << ' '
            << cfg.kernel_w << ' '
            << cfg.stride << ' '
            << cfg.padding << ' '
            << (cfg.apply_relu ? "relu" : "linear") << "\n";
        for (std::size_t oc = 0; oc < cfg.out_channels; ++oc) {
            for (std::size_t ic = 0; ic < cfg.in_channels; ++ic) {
                for (std::size_t kh = 0; kh < cfg.kernel_h; ++kh) {
                    for (std::size_t kw = 0; kw < cfg.kernel_w; ++kw) {
                        out << layer.weights[oc][ic][kh][kw];
                        if (kw + 1 != cfg.kernel_w) {
                            out << ' ';
                        }
                    }
                    out << "\n";
                }
            }
        }
        for (std::size_t oc = 0; oc < cfg.out_channels; ++oc) {
            out << layer.biases[oc];
            if (oc + 1 != cfg.out_channels) {
                out << ' ';
            }
        }
        out << "\n";
    }

    void operator()(const AvgPoolLayer &layer) const {
        out << "layer avgpool " << layer.kernel << ' ' << layer.stride << "\n";
    }

    void operator()(const FlattenLayer &) const {
        out << "layer flatten\n";
    }
};

} // namespace

void save_to_file(const Model &model, const std::string &path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("unable to open model output file: " + path);
    }

    if (model.input_kind == Model::InputKind::Vector) {
        out << "input vector " << model.vector_input_size << "\n";
    } else {
        out << "input image "
            << model.image_input_shape.channels << ' '
            << model.image_input_shape.height << ' '
            << model.image_input_shape.width << "\n";
    }

    for (const auto &layer : model.layers) {
        std::visit(LayerWriter{out}, layer);
    }
}

} // namespace gradient
