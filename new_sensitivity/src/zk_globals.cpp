#include "config_pc.hpp"
#include "gradient/zk_globals.hpp"

#include <vector>

unsigned long int mul_counter = 0;
std::vector<F> x_transcript;
std::vector<F> y_transcript;
F current_randomness = F_ZERO;

namespace gradient {

void initialize_kaizen_globals() {
    mul_counter = 0;
    x_transcript.clear();
    y_transcript.clear();
    current_randomness = F_ZERO;
}

} // namespace gradient
