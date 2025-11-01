#pragma once

#include <string>
#include <vector>

#include "CNN.h"
#include "GKR.h"
#include "config_pc.hpp"
#include "poly_commit.h"

struct _temp {
    std::vector<int> v;
};

extern int PC_scheme;
extern int Commitment_hash;
extern int levels;
extern double proving_time;
extern double total_time;
extern double range_proof_time;
extern int threads;
extern unsigned long int mul_counter;
extern double Forward_propagation;
extern double dx_computation;
extern double gradients_computation;
extern std::vector<int> predicates_size;
extern std::vector<struct proof> Transcript;
extern std::vector<F> SHA;
extern std::vector<F> H;
extern std::vector<F> x_transcript;
extern std::vector<F> y_transcript;
extern F current_randomness;
extern double aggregation_time;

void init_SHA();
std::vector<std::vector<F>> prepare_matrixes(std::vector<std::vector<F>> &M1,
                                             std::vector<std::vector<F>> &M2,
                                             std::vector<F> r1,
                                             std::vector<F> r2);
struct proof generate_4product_sumcheck_proof(std::vector<F> &v1,
                                              std::vector<F> &v2,
                                              F previous_r);
void extend_input(std::vector<F> input, std::vector<F> extended_input, int partitions);
std::vector<proof> mimc_sumcheck(std::vector<F> input);
struct proof generate_3product_sumcheck_proof(std::vector<F> &v1,
                                              std::vector<F> &v2,
                                              std::vector<F> &v3,
                                              F previous_r);
struct proof generate_2product_sumcheck_proof(std::vector<F> v1,
                                              std::vector<F> v2,
                                              F previous_r);
struct proof prove_ifft(std::vector<F> M);
struct proof prove_ifft_matrix(std::vector<std::vector<F>> M,
                               std::vector<F> r,
                               F previous_sum);
struct proof prove_fft(std::vector<F> M);
struct proof prove_fft_matrix(std::vector<std::vector<F>> M,
                              std::vector<F> r,
                              F previous_sum);
struct proof _prove_matrix2matrix(std::vector<std::vector<F>> M1,
                                  std::vector<std::vector<F>> M2,
                                  std::vector<F> r,
                                  F previous_sum);
struct proof prove_matrix2matrix(std::vector<std::vector<F>> M1,
                                 std::vector<std::vector<F>> M2);
std::vector<std::vector<F>> generate_bit_matrix(std::vector<F> bits, int domain);
void check_integrity(std::vector<F> bits, std::vector<F> num, std::vector<F> powers);
struct proof _prove_bit_decomposition(std::vector<F> bits,
                                      std::vector<F> r1,
                                      F previous_sum,
                                      int domain);
struct proof prove_bit_decomposition(std::vector<F> bits, std::vector<F> num, int domain);
F inner_product(std::vector<F> v1, std::vector<F> v2);
struct proof gkr_proof(std::string circuit_filename,
                       std::string data_filename,
                       std::vector<F> data,
                       std::vector<F> r,
                       bool debug);
std::vector<std::vector<F>> matrix2matrix(std::vector<std::vector<F>> M1,
                                          std::vector<std::vector<F>> M2);
struct _temp generate_vector();
std::vector<std::vector<std::vector<F>>> generate_matrix();
struct feedforward_proof insert_poly(std::string key,
                                     std::vector<F> poly,
                                     std::vector<F> eval_point,
                                     F eval,
                                     struct feedforward_proof P);
struct feedforward_proof insert_poly_only(std::string key,
                                          std::vector<F> poly,
                                          struct feedforward_proof P);
struct feedforward_proof update_poly(std::string key,
                                     std::vector<F> eval_point,
                                     F eval,
                                     struct feedforward_proof P);
std::vector<struct proof> prove_convolution(struct convolution_layer conv,
                                            std::vector<F> &r,
                                            F &previous_sum,
                                            bool avg);
void prove_division(std::vector<std::vector<F>> quotient,
                    std::vector<std::vector<F>> remainder,
                    std::vector<std::vector<F>> divident,
                    F divisor,
                    F e,
                    std::vector<F> &r,
                    F &previous_sum);
void prove_shift(std::vector<std::vector<F>> quotient,
                 std::vector<std::vector<F>> remainder,
                 std::vector<std::vector<F>> divident,
                 std::vector<F> &r,
                 F &previous_sum);
void prove_flattening(struct convolutional_network net, std::vector<F> &r, F &previous_sum);
void prove_avg(struct avg_layer avg_data, std::vector<F> &r, F &previous_sum, int pool_type);
void prove_max(std::vector<std::vector<F>> neg_input, std::vector<F> max_vals);
std::vector<F> find_max(std::vector<std::vector<float>> arr);
void prove_lookup(int N, int size);
std::vector<struct proof> prove_relu(struct relu_layer relu_data,
                                     std::vector<F> &r,
                                     F &previous_sum);
void prove_relu_backprop(struct relu_layer_backprop relu_data,
                         std::vector<F> &r,
                         F &previous_sum);
void prove_feedforward(struct convolutional_network net);
void prove_convolution_backprop(struct convolution_layer_backprop conv_back,
                                struct convolution_layer conv,
                                std::vector<F> &r,
                                F &previous_sum,
                                bool first);
void prove_avg_backprop(struct avg_layer_backprop avg_data,
                        struct convolution_layer conv,
                        struct convolution_layer_backprop conv_back,
                        std::vector<F> &r,
                        F &previous_sum,
                        bool final_avg);
void prove_correct_gradient_computation(struct convolution_layer_backprop conv_back,
                                        struct convolution_layer conv,
                                        std::vector<F> &r,
                                        F &previous_sum,
                                        bool avg);
void prove_dense_backprop(struct dense_layer_backprop dense_backprop,
                          std::vector<F> &r,
                          F &previous_sum);
void prove_dense_gradient_computation(struct dense_layer_backprop dense_backprop,
                                      struct fully_connected_layer dense,
                                      std::vector<F> &r,
                                      F &previous_sum);
void flat_layer(struct convolutional_network net,
                struct convolution_layer_backprop conv_back,
                struct relu_layer_backprop relu_backprop,
                std::vector<F> &r,
                F &previous_sum);
void prove_backprop(struct convolutional_network net);
std::vector<std::vector<F>> prepare_input(std::vector<std::vector<F>> input);
void check_dataset(int batch, int input_dim);
void get_witness(struct convolutional_network net, std::vector<F> &witness);
void clear_witness(struct convolutional_network &net);
void clear_model(struct convolutional_network &net);
void get_model(struct convolutional_network net, std::vector<F> &model);
std::vector<int> get_sizes(struct convolutional_network net);
void aggregate_commited_data(std::vector<std::vector<F>> polynomials,
                             std::vector<std::vector<F>> old_polynomials);
std::vector<F> rotate(std::vector<F> bits, int shift);
std::vector<F> shift(std::vector<F> bits, int shift);
SHA_witness get_sha_witness(std::vector<F> words);
std::vector<F> prove_aggregation(aggregation_witness data, int level);
std::vector<F> prove_aggr(std::vector<std::vector<std::vector<std::vector<F>>>> matrixes,
                          std::vector<std::vector<commitment>> comm);
void test_aggregation(int level, int bitlen);
void reduce_polynomials(std::vector<F> poly1, std::vector<F> poly2);
