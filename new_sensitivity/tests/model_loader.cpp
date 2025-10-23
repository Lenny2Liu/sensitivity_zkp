#include "gradient/model.hpp"
#include "gradient/zk_globals.hpp"

#include <iostream>

int main() {
    gradient::initialize_kaizen_globals();
    try {
        auto model = gradient::load_from_file("../examples/lenet_small_model.txt");
        std::cout << "layers: " << model.layers.size() << std::endl;
        std::cout << "input kind: " << (model.input_kind == gradient::Model::InputKind::Image ? "image" : "vector") << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
