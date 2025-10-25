#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "CNN.h"
#include "GKR.h"
#include "quantization.h"
#include "utils.hpp"

using namespace std;

extern int batch;
extern vector<struct proof> Transcript;

// Backprop primitives implemented in CNN.cpp.
struct dense_layer_backprop dense_backprop(vector<vector<F>> &dx, struct fully_connected_layer dense);
struct relu_layer_backprop relu_backprop(vector<F> &dx, struct relu_layer relu);
struct convolution_layer_backprop conv_backprop(vector<vector<vector<vector<F>>>> &derr,
                                                int &dx_width,
                                                struct convolution_layer conv,
                                                vector<vector<vector<vector<F>>>> rotated_Filter);
struct avg_layer_backprop avg_pool_der(vector<vector<vector<vector<F>>>> &derr,
                                       int dx_width,
                                       int old_w);
vector<vector<vector<vector<F>>>> batch_convolution_der(vector<vector<vector<vector<F>>>> input,
                                                        vector<vector<vector<vector<F>>>> derr,
                                                        vector<vector<vector<vector<F>>>> w);
void prove_backprop(struct convolutional_network net);
struct proof _prove_bit_decomposition(vector<F> bits, vector<F> r1, F previous_sum, int domain);

static constexpr int MODEL_LENET = 1;

struct SensitivityProofResult {
    vector<F> gradient_column;
    F gradient_norm_sq = F_ZERO;
    F bound_sq = F_ZERO;
    float bound_value = 0.0f;
    vector<vector<struct proof>> gradient_proofs;
    vector<struct proof> norm_bound_proof;
};

namespace {

struct GradientComputation {
    convolutional_network annotated_net;
    vector<vector<vector<vector<F>>>> gradient_tensor;
    vector<F> gradient_flat;
};

int final_output_dimension(const convolutional_network &net) {
    if (net.fully_connected.empty()) {
        throw runtime_error("Network has no dense layers; unable to compute sensitivity");
    }
    if (net.fully_connected.back().Z_new.empty()) {
        throw runtime_error("Final dense layer has empty output");
    }
    return static_cast<int>(net.fully_connected.back().Z_new[0].size());
}

void reset_backprop_state(convolutional_network &net) {
    net.fully_connected_backprop.clear();
    net.relus_backprop.clear();
    net.avg_backprop.clear();
    net.convolutions_backprop.clear();
    net.der.clear();
    net.w.clear();
    net.der_dim.clear();
}

int input_feature_count(const vector<vector<vector<vector<F>>>> &input_tensor) {
    long long total = 0;
    for (const auto &batch_item : input_tensor) {
        for (const auto &channel : batch_item) {
            for (const auto &row : channel) {
                total += static_cast<long long>(row.size());
            }
        }
    }
    return static_cast<int>(total);
}

double approximate_real_value(const F &value, int fractional_bits = Q) {
    long double scaled = static_cast<long double>(value.toint128());
    // cout << "[debug] approximate_real_value: scaled=" << scaled << ", fractional_bits=" << fractional_bits << endl;
    long double denom = std::ldexp(1.0L, fractional_bits);
    scaled /= denom;
    scaled /= 16;
    // cout << "[debug] approximate_real_value: result=" << static_cast<double>(scaled) << endl;
    cout << "[debug] approximate_real_value: scaled=" << scaled / denom << ", denom=" << denom << endl;
    return static_cast<double>(scaled / denom);
}

void ensure_bound_above_norm(F &bound, F &bound_sq, const F &norm_sq, float &bound_value) {
    __int128 norm_int = norm_sq.toint128();
    __int128 bound_int = bound_sq.toint128();
    while (norm_int > bound_int) {
        bound_value *= 2.0f;
        bound = quantize(bound_value);
        bound_sq = bound * bound;
        bound_int = bound_sq.toint128();
    }
}

vector<F> flatten_matrix(const vector<vector<F>> &matrix) {
    vector<F> flattened;
    for (const auto &row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

void dump_gradient(const string &label, int output_index, const vector<F> &gradient) {
	std::ofstream out_file(label + ".txt");
	cerr << "[grad][" << label << "][i=" << output_index << "] size=" << gradient.size() << '\n';
	for (size_t idx = 0; idx < gradient.size(); ++idx) {
		long long raw = static_cast<long long>(gradient[idx].toint128());
		cerr << "  [" << idx << "]=" << raw << '\n';
		
		if (out_file.is_open()) {
			out_file << "[" << idx << "]=" << raw << '\n';
		}
	}
	if (out_file.is_open()) {
		out_file.close();
	} else {
		cerr << "Failed to open file for dumping gradient: " << label << ".txt\n";
	}
}

void dump_tensor_gradient(const string &label,
                          int output_index,
                          const vector<vector<vector<vector<F>>>> &tensor) {
	std::ofstream out_file(label + ".txt");
	if (out_file.is_open()) {
		out_file << "[grad][" << label << "][i=" << output_index << "] size=" << tensor2vector(tensor).size() << '\n';
		const auto flat = tensor2vector(tensor);
		for (size_t idx = 0; idx < flat.size(); ++idx) {
			long long raw = static_cast<long long>(flat[idx].toint128());
			out_file << "  [" << idx << "]=" << raw << '\n';
		}
		out_file.close();
	} else {
		cerr << "Failed to open file for dumping gradient: " << label << ".txt\n";
	}
}

GradientComputation compute_input_gradient(convolutional_network net,
                                           int output_index,
                                           const vector<vector<vector<vector<F>>>> &original_input,
                                           int sample_index = 0) {
    if (batch != 1) {
        throw runtime_error("Sensitivity proof currently supports batch size 1 only");
    }
    if (original_input.empty()) {
        throw runtime_error("Original input tensor is empty");
    }

    GradientComputation result;
    reset_backprop_state(net);

    const int dense_outputs = final_output_dimension(net);
    if (output_index < 0 || output_index >= dense_outputs) {
        throw runtime_error("Output index out of range");
    }
    if (sample_index < 0 || sample_index >= batch) {
        throw runtime_error("Sample index out of range");
    }
    // back_propagation(net);  // Ensure forward pass is done


    vector<vector<F>> out_der(batch);
    for (int b = 0; b < batch; ++b) {
        out_der[b].assign(dense_outputs, F_ZERO);
    }
    out_der[sample_index][output_index] = quantize(1.0f);
    dump_gradient("dense_init", output_index, flatten_matrix(out_der));
	struct convolution_layer_backprop conv_der;
	struct dense_layer_backprop dense_der;
	struct relu_layer_backprop relu_der;
	int relu_counter = net.relus.size()-1;
	int in_size;
	vector<vector<vector<vector<F>>>> der(batch),dx(batch);
	vector<vector<F>> dense_dx(batch);
	// virgo::printNested(out_der, std::cout);
	// std::cout << std::endl;
	for(int i = net.Weights.size()-1; i >= 0; i--){
		dense_der = dense_backprop(out_der,net.fully_connected[i]);
		net.fully_connected_backprop.push_back(dense_der);
		vector<F> v = convert2vector(out_der);
		relu_der = relu_backprop(v,net.relus[relu_counter]);
		for(int j = 0; j < out_der.size(); j++){
			for(int k = 0; k < out_der[j].size(); k++){
				out_der[j][k] = v[j*out_der[j].size() + k];
			}
		}
		net.relus_backprop.push_back(relu_der);
		in_size = net.Weights[i][0].size();
		relu_counter--;
	}

	int last_conv = net.Filters.size()-1;
	for(int i = 0; i < der.size(); i++){
		der[i].resize(net.final_out);
		for(int j = 0; j < der[i].size(); j++){
			der[i][j].resize(net.flatten_n);
			for(int k = 0; k < der[i][j].size(); k++){
				for(int l = 0; l < der[i][j].size(); l++){
					der[i][j][k].push_back(F(0));
				}
			}
			for(int k = 0; k < net.final_w; k++){
				for(int l = 0; l < net.final_w; l++){
					der[i][j][k][l] = out_der[i][net.final_w*net.final_w*j + k*net.final_w + l];
				}
			}
		}
	}

	int real_dx_width = net.final_w; 
	if(net.convolution_pooling[net.Filters.size() - 1] != 0){
		printf("Avg Pooling backprop for last layer==============================\n");
		net.avg_backprop.push_back(avg_pool_der(der,real_dx_width,net.avg_layers[net.Filters.size() - 1].n));
		real_dx_width *=2;
	}
	for(int i = net.Filters.size() - 1; i >= 0; i--){
		
		net.convolutions_backprop.push_back(conv_backprop(der,real_dx_width,net.convolutions[i],net.Rotated_Filters[i]));
		if(i != 0){
			if(net.convolution_pooling[i-1] != 0){
				net.avg_backprop.push_back(avg_pool_der(der,real_dx_width,net.avg_layers[i-1].n));
				real_dx_width *= 2;
			}
			
			int w = der[0][0].size();
			net.der_dim.push_back(der.size()*der[0].size()*w*w);
			if(der.size()*der[0].size()*w*w != net.relus[relu_counter].most_significant_bits.size()){
				vector<vector<vector<vector<F>>>> temp(der.size());
				net.der.push_back(der);

				w = w/2;
				net.w.push_back(w);
				for(int j = 0; j < temp.size(); j++){
					temp[j].resize(der[j].size());
					for(int k = 0; k < temp[j].size(); k++){
						temp[j][k].resize(w);
						for(int l = 0; l < w; l++){
							temp[j][k][l].resize(w);
							for(int m = 0; m < w; m++){
								temp[j][k][l][m] = der[j][k][l][m];
							}
						}
					}
				}
				der = temp;
			}
			vector<F> v = tensor2vector(der);
			relu_der = relu_backprop(v,net.relus[relu_counter]);
			der = vector2tensor(relu_der.dx,der,w);
			net.relus_backprop.push_back(relu_der);
			relu_counter--;
		}
	}

    vector<vector<vector<vector<F>>>> input_dx = batch_convolution_der(original_input, der, net.Rotated_Filters[0]);
    std::cout << net.convolutions_backprop.size() << " convolution backprop layers processed.\n";
    result.annotated_net = std::move(net);
    result.gradient_tensor = std::move(input_dx);
    result.gradient_flat = tensor2vector(result.gradient_tensor);
    return result;
}

} // namespace

SensitivityProofResult prove_model_sensitivity(convolutional_network net,
                                               const vector<vector<vector<vector<F>>>> &input_tensor,
                                               int feature_index,
                                               float initial_bound) {
    if (feature_index < 0) {
        throw runtime_error("Feature index must be non-negative");
    }
    SensitivityProofResult result;
    const int total_features = input_feature_count(input_tensor);
    if (feature_index >= total_features) {
        throw runtime_error("Feature index exceeds input dimensionality");
    }
    feature_index = 199;
    const int outputs = final_output_dimension(net);
    vector<F> gradient_column(outputs, F_ZERO);
    result.gradient_proofs.reserve(outputs);
    // outputs = 1;
    for (int out_idx = 0; out_idx < outputs; ++out_idx) {

		vector<vector<vector<vector<F>>>> dX;
		// convolutional_network annotated = back_propagation_sens(net, &dX, out_idx, 0);
		GradientComputation computation =
            compute_input_gradient(net, out_idx, input_tensor);


		// vector<F> flat = tensor2vector(dX);     // flatten [B][C][H][W]
		gradient_column[out_idx] = computation.gradient_flat[feature_index];  // pick the feature of interest
		Transcript.clear();
        // Print the gradient column for this output
        cout << "Gradient column for output " << out_idx << ":" << endl;
        for (size_t i = 0; i < computation.gradient_flat.size(); i++) {
            if (i == feature_index) {
                cout << "  [" << i << "] = " << approximate_real_value(computation.gradient_flat[i])
                     << " (raw=" << static_cast<long long>(computation.gradient_flat[i].toint128()) << ")"
                     << " <-- selected feature" << endl;
            }
        }
		prove_backprop(computation.annotated_net);
		result.gradient_proofs.push_back(Transcript);
		Transcript.clear();
    }

    result.gradient_column = std::move(gradient_column);

    F norm_sq = F_ZERO;
    for (const F &component : result.gradient_column) {
        norm_sq += component * component;
    }
    result.gradient_norm_sq = norm_sq;

    float bound_value = std::max(initial_bound, 1.0f);
    F bound = quantize(bound_value);
    F bound_sq = bound * bound;
    ensure_bound_above_norm(bound, bound_sq, norm_sq, bound_value);

    result.bound_value = bound_value;
    result.bound_sq = bound_sq;

    vector<F> range_proof_data = {norm_sq, bound_sq - norm_sq};
    vector<F> randomness = generate_randomness(1, F_ZERO);
    F eval = evaluate_vector(range_proof_data, randomness);
    vector<F> bits = prepare_bit_vector(range_proof_data, 64);

    Transcript.clear();
    Transcript.push_back(_prove_bit_decomposition(bits, randomness, eval, 64));
    result.norm_bound_proof = Transcript;
    Transcript.clear();

    return result;
}

void run_sensitivity_proof_demo() {
    const int batch_size = 1;
    const int channels = 1;
    const int feature_index = 0;
    const float initial_bound = 64.0f;

    convolutional_network net = init_network(MODEL_LENET, batch_size, channels);
    vector<vector<vector<vector<F>>>> input_tensor;
    net = feed_forward(input_tensor, net, channels);

    SensitivityProofResult proof =
        prove_model_sensitivity(net, input_tensor, feature_index, initial_bound);

    double total_backprop_kb = 0.0;
    for (const auto &proofs : proof.gradient_proofs) {
        total_backprop_kb += proof_size(proofs);
    }
    double norm_bound_kb = proof_size(proof.norm_bound_proof);

    cout << "Sensitivity gradient column for input feature " << feature_index << '\n';
    for (int i = 0; i < proof.gradient_column.size(); ++i) {
        long long raw = static_cast<long long>(proof.gradient_column[i].toint128());
        cout << "  dy[" << i << "]/dx[" << feature_index << "] ≈ "
             << approximate_real_value(proof.gradient_column[i]) << " (raw=" << raw << ")\n";
    }
    cout << "Squared norm ≈ " << approximate_real_value(proof.gradient_norm_sq, 2 * Q)
         << " (bound ≈ " << approximate_real_value(proof.bound_sq, 2 * Q) << ")\n";
    cout << "Backprop proofs total size ≈ " << total_backprop_kb << " KB\n";
    cout << "Norm bound proof size ≈ " << norm_bound_kb << " KB\n";
}
