#pragma once

#include "GKR.h"
#include "gradient/model.hpp"
#include "gradient/pipeline.hpp"

#include <string>
#include <vector>

namespace gradient {

std::vector<LayerProof> prove_dense_layers(const Model &model,
                                           const GradientComputationResult &result);

} // namespace gradient
