#include "gradient/gkr_plan.hpp"

#include <sstream>

namespace gradient {
namespace {

std::string layer_label(const LayerVariant &layer, std::size_t index) {
    std::ostringstream oss;
    oss << index << ": "
        << [&]() {
               if (const auto *dense = std::get_if<DenseLayer>(&layer)) {
                   std::ostringstream tmp;
                   tmp << "dense(" << dense->config.in_features << "→" << dense->config.out_features
                       << ", " << (dense->config.apply_relu ? "relu" : "linear") << ")";
                   return tmp.str();
               }
               if (const auto *conv = std::get_if<Conv2DLayer>(&layer)) {
                   std::ostringstream tmp;
                   tmp << "conv2d(" << conv->config.in_channels << "→" << conv->config.out_channels
                       << ", k=" << conv->config.kernel_h << "x" << conv->config.kernel_w
                       << ", s=" << conv->config.stride
                       << ", p=" << conv->config.padding
                       << ", " << (conv->config.apply_relu ? "relu" : "linear") << ")";
                   return tmp.str();
               }
               if (std::holds_alternative<AvgPoolLayer>(layer)) {
                   const auto &pool = std::get<AvgPoolLayer>(layer);
                   std::ostringstream tmp;
                   tmp << "avgpool(k=" << pool.kernel << ", s=" << pool.stride << ")";
                   return tmp.str();
               }
               return std::string("flatten");
           }();
    return oss.str();
}

} // namespace

std::vector<GKRInvocationPlan> build_gkr_plan(const Model &model,
                                              const GradientComputationResult &result) {
    std::vector<GKRInvocationPlan> plan;
    plan.reserve(model.layers.size());

    for (std::size_t idx = 0; idx < model.layers.size(); ++idx) {
        const auto &layer = model.layers[idx];
        GKRInvocationPlan entry;
        entry.layer_label = layer_label(layer, idx);

        std::ostringstream desc;
        if (const auto *dense = std::get_if<DenseLayer>(&layer)) {
            desc << "Feed-forward dot products and bias additions already covered; for backward pass use KAIZEN"
                 << " dense routines (`prove_dx_prod`, `prove_dw_prod`) with in_dim=" << dense->config.in_features
                 << " and out_dim=" << dense->config.out_features << ".";
        } else if (const auto *conv = std::get_if<Conv2DLayer>(&layer)) {
            const auto &cache = std::get<ConvLayerCache>(result.forward_context.caches[idx]);
            desc << "Use convolution backprop circuits (`conv_backprop`, `prove_dx_prod`/`prove_dw_prod` equivalents)"
                 << " with in_shape=" << cache.input_shape.channels << "x" << cache.input_shape.height << "x"
                 << cache.input_shape.width << ", kernel=" << conv->config.kernel_h << "x"
                 << conv->config.kernel_w << ", out_shape=" << cache.output_shape.channels << "x"
                 << cache.output_shape.height << "x" << cache.output_shape.width << ".";
        } else if (std::holds_alternative<AvgPoolLayer>(layer)) {
            const auto &pool = std::get<AvgPoolLayer>(layer);
            desc << "Reuse KAIZEN avg-pooling derivative circuit (`prove_avg_der`) with kernel=" << pool.kernel
                 << " stride=" << pool.stride << ".";
        } else {
            desc << "Flatten is a permutation; prove via copy constraints or hash-commitment linking.";
        }

        plan.push_back({entry.layer_label, desc.str()});
    }

    return plan;
}

} // namespace gradient
