#pragma once

#include "gradient/forward_backward.hpp"
#include "gradient/model.hpp"
#include "gradient/pipeline.hpp"

#include <string>
#include <vector>

namespace gradient {

struct GKRInvocationPlan {
    std::string layer_label;
    std::string description;
};

std::vector<GKRInvocationPlan> build_gkr_plan(const Model &model,
                                              const GradientComputationResult &result);

} // namespace gradient
