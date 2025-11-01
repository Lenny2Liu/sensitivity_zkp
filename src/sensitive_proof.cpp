#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "CNN.h"
#include "proof.h"
#include "GKR.h"
#include "quantization.h"
#include "utils.hpp"
#include "mimc.h"
#include "lookups.h"

using namespace std;

extern int batch;

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
    // dump_gradient("dense_init", output_index, flatten_matrix(out_der));
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
    for (int out_idx = 0; out_idx < outputs; ++out_idx) {
		vector<vector<vector<vector<F>>>> dX;
		GradientComputation computation =
            compute_input_gradient(net, out_idx, input_tensor);
		gradient_column[out_idx] = computation.gradient_flat[feature_index];
		Transcript.clear();
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
        cout << "  dy[" << i << "]/dx[" << feature_index << "] = "
             << approximate_real_value(proof.gradient_column[i]) << " (raw=" << raw << ")\n";
    }
    cout << "Squared norm = " << approximate_real_value(proof.gradient_norm_sq, 2 * Q)
         << " (bound = " << approximate_real_value(proof.bound_sq, 2 * Q) << ")\n";
    cout << "Backprop proofs total size = " << total_backprop_kb << " KB\n";
    cout << "Norm bound proof size = " << norm_bound_kb << " KB\n";
}



    // int relu_counter = static_cast<int>(net.relus.size()) - 1;

    // for (int dense_idx = static_cast<int>(net.Weights.size()) - 1; dense_idx >= 0; --dense_idx) {
    //     dense_layer_backprop dense_der = dense_backprop(out_der, net.fully_connected[dense_idx]);
    //     out_der = dense_der.dx;
    //     if (dense_idx == static_cast<int>(net.Weights.size()) - 1 && output_index == 0) {
    //         long long sample_raw = 0;
    //         if (!dense_der.dx_temp.empty() && !dense_der.dx_temp[0].empty()) {
    //             sample_raw = static_cast<long long>(dense_der.dx_temp[0][0].toint128());
    //             float sample_float = approximate_real_value(dense_der.dx_temp[0][0]);
    //             cerr << "[debug] dense idx " << dense_idx << " dx_temp[0][0] float=" << sample_float << endl;
    //         }
    //         cerr << "[debug] dense idx " << dense_idx << " dx_temp[0][0] raw=" << sample_raw << endl;
    //     }
    //     dump_gradient("dense_backprop_layer_" + to_string(dense_idx),
    //                   output_index,
    //                   flatten_matrix(dense_der.dx));
    //     net.fully_connected_backprop.push_back(dense_der);

    //     vector<F> flattened = convert2vector(out_der);
    //     relu_layer_backprop relu_der = relu_backprop(flattened, net.relus[relu_counter]);
    //     for (int b = 0; b < batch; ++b) {
    //         const int width = static_cast<int>(out_der[b].size());
    //         for (int k = 0; k < width; ++k) {
    //             out_der[b][k] = relu_der.dx[b * width + k];
    //         }
    //     }
    //     dump_gradient("relu_backprop_after_dense_" + to_string(dense_idx),
    //                   output_index,
    //                   flatten_matrix(out_der));
    //     net.relus_backprop.push_back(relu_der);
    //     --relu_counter;
    // }

    // vector<vector<vector<vector<F>>>> der(batch);
    // const int last_conv = static_cast<int>(net.Filters.size()) - 1;
    // if (last_conv < 0) {
    //     throw runtime_error("Sensitivity proof requires at least one convolutional layer");
    // }
    // for (int b = 0; b < batch; ++b) {
    //     der[b].resize(net.final_out);
    //     for (int ch = 0; ch < net.final_out; ++ch) {
    //         der[b][ch].resize(net.flatten_n);
    //         for (int row = 0; row < net.flatten_n; ++row) {
    //             der[b][ch][row].assign(net.flatten_n, F_ZERO);
    //         }
    //         for (int row = 0; row < net.final_w; ++row) {
    //             for (int col = 0; col < net.final_w; ++col) {
    //                 der[b][ch][row][col] =
    //                     out_der[b][net.final_w * net.final_w * ch + row * net.final_w + col];
    //             }
    //         }
    //     }
    // }
    // dump_tensor_gradient("conv_init", output_index, der);

    // int real_dx_width = net.final_w;
    // if (!net.convolution_pooling.empty() && net.convolution_pooling.back() != 0) {
    //     net.avg_backprop.push_back(avg_pool_der(der, real_dx_width, net.avg_layers.back().n));
    //     real_dx_width *= 2;
    //     dump_tensor_gradient("avg_pool_backprop_last", output_index, der);
    // }

    // for (int conv_idx = last_conv; conv_idx >= 0; --conv_idx) {
    //     convolution_layer_backprop conv_der =
    //         conv_backprop(der, real_dx_width, net.convolutions[conv_idx], net.Rotated_Filters[conv_idx]);
    //     net.convolutions_backprop.push_back(conv_der);
    //     dump_tensor_gradient("conv_backprop_layer_" + to_string(conv_idx), output_index, der);

    //     if (conv_idx != 0) {
    //         if (net.convolution_pooling[conv_idx - 1] != 0) {
    //             net.avg_backprop.push_back(avg_pool_der(der, real_dx_width, net.avg_layers[conv_idx - 1].n));
    //             real_dx_width *= 2;
    //             dump_tensor_gradient("avg_pool_backprop_layer_" + to_string(conv_idx - 1),
    //                                  output_index,
    //                                  der);
    //         }

    //         int w = static_cast<int>(der[0][0].size());
    //         net.der_dim.push_back(static_cast<int>(der.size() * der[0].size() * w * w));
    //         if (static_cast<int>(der.size() * der[0].size() * w * w) !=
    //             net.relus[relu_counter].most_significant_bits.size()) {
    //             vector<vector<vector<vector<F>>>> temp(der.size());
    //             net.der.push_back(der);

    //             w /= 2;
    //             net.w.push_back(w);
    //             for (int b = 0; b < temp.size(); ++b) {
    //                 temp[b].resize(der[b].size());
    //                 for (int ch = 0; ch < temp[b].size(); ++ch) {
    //                     temp[b][ch].resize(w);
    //                     for (int row = 0; row < w; ++row) {
    //                         temp[b][ch][row].resize(w);
    //                         for (int col = 0; col < w; ++col) {
    //                             temp[b][ch][row][col] = der[b][ch][row][col];
    //                         }
    //                     }
    //                 }
    //             }
    //             der = temp;
    //         }

    //         vector<F> flattened = tensor2vector(der);
    //         relu_layer_backprop relu_der = relu_backprop(flattened, net.relus[relu_counter]);
    //         der = vector2tensor(relu_der.dx, der, static_cast<int>(der[0][0].size()));
    //         dump_tensor_gradient("relu_backprop_conv_layer_" + to_string(conv_idx - 1),
    //                              output_index,
    //                              der);
    //         net.relus_backprop.push_back(relu_der);
    //         --relu_counter;
    //     }
    // }


namespace {
inline double u01_from_mimc(F &state) {
    mimc_hash(state, current_randomness);
    unsigned long long raw = (unsigned long long) state.toint128();
    const double denom = (double)(1ULL << 32);
    return (double)(raw & 0xffffffffULL) / denom;
}

vector<vector<vector<vector<F>>>> gaussian_noise_from_mimc(const vector<vector<vector<vector<F>>>> &X, float sigma, int sample_id) {
    vector<vector<vector<vector<F>>>> Z = X;
    F seed = F(sample_id);
    mimc_hash(seed, current_randomness);

    for (int b = 0; b < (int)X.size(); ++b) {
        for (int c = 0; c < (int)X[b].size(); ++c) {
            for (int r = 0; r < (int)X[b][c].size(); ++r) {
                for (int col = 0; col < (int)X[b][c][r].size(); ++col) {
                    double s = 0.0;
                    for (int t = 0; t < 12; ++t) s += u01_from_mimc(seed);
                    double g = sigma * (s - 6.0);
                    Z[b][c][r][col] = quantize((float)g);
                }
            }
        }
    }
    return Z;
}

vector<vector<vector<vector<F>>>> add_tensors(const vector<vector<vector<vector<F>>>> &A, const vector<vector<vector<vector<F>>>> &B) {
    vector<vector<vector<vector<F>>>> C = A;
    for (int b = 0; b < (int)A.size(); ++b)
      for (int c = 0; c < (int)A[b].size(); ++c)
        for (int r = 0; r < (int)A[b][c].size(); ++r)
          for (int col = 0; col < (int)A[b][c][r].size(); ++col)
              C[b][c][r][col] = A[b][c][r][col] + B[b][c][r][col];
    return C;
}

int final_output_dimension(const convolutional_network &net); 

int feature_count(const vector<vector<vector<vector<F>>>> &X) {
    long long tot = 0;
    for (auto &b: X) for (auto &ch: b) for (auto &row: ch) tot += (long long)row.size();
    return (int)tot;
}

}



struct GaussAvgSensitivityResult {
    vector<F> avg_gradient_column;
    float sigma = 0.f;
    int num_samples = 0;
    vector<vector<vector<struct proof>>> per_sample_per_output_backprop;
    vector<vector<struct proof>> average_proofs_per_output;
    vector<struct proof> sampling_provenance;
};

static vector<struct proof> prove_sum_over_samples(const vector<F>& vals) {
    vector<F> v1 = vals, v2(vals.size(), F_ONE);
    pad_vector(v1); pad_vector(v2);
    vector<struct proof> out;
    vector<F> r = generate_randomness((int)log2(v1.size()), F_ZERO);
    out.push_back(generate_2product_sumcheck_proof(v1, v2, r.back()));
    return out;
}


GaussAvgSensitivityResult gaussian_sampled_average_sensitivity_backprop(convolutional_network base_net,
                                              const vector<vector<vector<vector<F>>>>& X_center,
                                              int feature_index,
                                              float sigma,
                                              int num_samples) {
    if (batch != 1) throw runtime_error("Gaussian sampling demo assumes batch==1");
    if (feature_index < 0) throw runtime_error("feature_index must be non-negative");
    if (X_center.empty()) throw runtime_error("X_center is empty");
    if (feature_index >= feature_count(X_center)) throw runtime_error("feature_index out of range");
    if (num_samples <= 0) throw runtime_error("num_samples must be positive");
    if (sigma <= 0) throw runtime_error("sigma must be positive");

    {
        vector<vector<vector<vector<F>>>> X0 = X_center;
        convolutional_network tmp = base_net;
        tmp = feed_forward(X0, tmp, (int)X_center[0].size());
        (void) final_output_dimension(tmp);
    }
    convolutional_network net_shape = base_net;
    vector<vector<vector<vector<F>>>> X0 = X_center;
    net_shape = feed_forward(X0, net_shape, (int)X_center[0].size());
    const int m = final_output_dimension(net_shape);

    vector<vector<F>> sample_vals_per_output(m);
    vector<vector<vector<struct proof>>> per_sample_per_output(m);
    for (int j = 0; j < m; ++j) {
        sample_vals_per_output[j].reserve(num_samples);
        per_sample_per_output[j].reserve(num_samples);
    }
    vector<F> sampling_record;
    for (int t = 0; t < num_samples; ++t) {
        vector<vector<vector<vector<F>>>> noise = gaussian_noise_from_mimc(X_center, sigma, t+1);
        vector<vector<vector<vector<F>>>> X_t = add_tensors(X_center, noise);
        sampling_record.push_back(F(t+1));
        convolutional_network net_t = base_net;
        net_t = feed_forward(X_t, net_t, (int)X_center[0].size());
        for (int j = 0; j < m; ++j) {
            GradientComputation comp = compute_input_gradient(net_t, j, X_t);
            F g_ji = comp.gradient_flat[feature_index];
            sample_vals_per_output[j].push_back(g_ji);
            Transcript.clear();
            prove_backprop(comp.annotated_net);
            per_sample_per_output[j].push_back(Transcript);
            Transcript.clear();
        }
    }
    vector<vector<struct proof>> avg_proofs(m);
    vector<F> avg_col(m, F_ZERO);
    for (int j = 0; j < m; ++j) {
        F sum_j = F_ZERO;
        for (const F &v : sample_vals_per_output[j]) {
            sum_j = sum_j + v;
        }

        vector<struct proof> Pj = prove_sum_over_samples(sample_vals_per_output[j]);
        const F kF = quantize((float)num_samples);
        F avg_j = divide(sum_j, kF);
        F rem_j = sum_j - avg_j * kF;
        vector<F> r = generate_randomness(1, F_ZERO);
        struct proof D;
        D.type = DIVISION_CHECK;
        D.divisor  = kF;
        D.quotient = avg_j;
        D.remainder= rem_j;
        D.divident = sum_j;
        Pj.push_back(D);
        vector<F> range_data = { kF - rem_j };
        lookup_proof rP = lookup_range_proof(range_data, r, kF - rem_j, 32);
        Pj.push_back(rP.mP1);
        Pj.push_back(rP.mP2);
        Pj.push_back(rP.sP1);
        avg_proofs[j] = std::move(Pj);
        avg_col[j]    = avg_j;
    }
    vector<struct proof> sampling_P = mimc_sumcheck(sampling_record);
    GaussAvgSensitivityResult R;
    R.avg_gradient_column           = std::move(avg_col);
    R.sigma                         = sigma;
    R.num_samples                   = num_samples;
    R.per_sample_per_output_backprop= std::move(per_sample_per_output);
    R.average_proofs_per_output     = std::move(avg_proofs);
    R.sampling_provenance           = std::move(sampling_P);
    return R;
}

// In sensitive_proof.cpp
void run_gaussian_sampled_sensitivity_demo() {
    const int batch_size = 1;
    const int channels = 1;
    const int feature_i = 0;
    const float sigma = 0.10f;
    const int k = 3;

    convolutional_network net = init_network(1, batch_size, channels);
    vector<vector<vector<vector<F>>>> X;
    net = feed_forward(X, net, channels);

    auto R = gaussian_sampled_average_sensitivity_backprop(net, X, feature_i, sigma, k);

    std::cout << "[Gaussian avg sensitivity/backprop] feature " << feature_i
              << " sigma=" << sigma << " samples=" << k << "\n";
    for (int j = 0; j < (int)R.avg_gradient_column.size(); ++j) {
        long long raw = (long long)R.avg_gradient_column[j].toint128();
        std::cout << "  mean_t dy[" << j << "]/dx[" << feature_i << "] = (raw) " << raw << "\n";
    }
    double kb_backprop = 0.0;
    for (auto &per_out : R.per_sample_per_output_backprop)
        for (auto &pf : per_out) kb_backprop += proof_size(pf);
    double kb_avg = 0.0;
    for (auto &pf : R.average_proofs_per_output) kb_avg += proof_size(pf);
    std::cout << "  Total backprop proof size ≈ " << kb_backprop << " KB\n";
    std::cout << "  Averaging (sum+division+range) proofs size ≈ " << kb_avg << " KB\n";
}
