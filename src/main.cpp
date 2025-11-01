#include <stdio.h>
//#include <mcl/bn256.hpp>
//#include <mcl/bls12_381.hpp>
//#include <mcl/bn.h>
#include <vector>
#include <polynomial.h>
#include <math.h>
#include "MLP.h"
#include <stdlib.h>
#include <string.h>
//#include <gmp.h>
#include <time.h>
#include "mimc.h"
#include "quantization.h"
#include "GKR.h"
#include <time.h>
#include <chrono>
#include "utils.hpp"
#include "pol_verifier.h"
#include "CNN.h"
#include "elliptic_curves.h"
#include "config_pc.hpp"
#include "poly_commit.h"
#include "lookups.h"
#include<unistd.h>
#include "proof.h"



void run_sensitivity_proof_demo();
void run_gaussian_sampled_sensitivity_demo();
// void run_integrated_gradients_demo();

extern int partitions;

using namespace std;

int main(int argc, char *argv[]){
	
	elliptic_curves_init();
	init_hash();
    init_SHA();
	PC_scheme = 2;
	Commitment_hash = 1;
	levels = 1;
	

	
	//arr.clear();
	
	int input_dim;
    
	char buff[257];
   	vector<vector<vector<vector<F>>>> X;
   	vector<F> r;
   	int model;

    if(argc >= 2 && strcmp(argv[1],"S") == 0){
        run_sensitivity_proof_demo();
        return 0;
    }
	if (argc >= 2 && strcmp(argv[1], "SM") == 0) {
		run_gaussian_sampled_sensitivity_demo();
		return 0;
	}
	// if (argc >= 2 && strcmp(argv[1], "SIG") == 0) {
	// 	run_integrated_gradients_demo();
	// 	return 0;
	// }
   	if(strcmp(argv[1],"LENET") == 0){
   		input_dim = 32;
   		printf("Lenet\n");
   		model = 1;
   	}
   	else if(strcmp(argv[1],"AlexNet") == 0){
   		input_dim = 64;
   		printf("AlexNet\n");
   		model = 4;
   	}
   	else if(strcmp(argv[1],"mAlexNet") == 0){
   		input_dim = 64;
   		printf("mAlexNet\n");
   		model = 5;
   	}
   	else if(strcmp(argv[1],"VGG") == 0){
   		input_dim = 64;
   		model = 2;
   	}
   	else if(strcmp(argv[1],"TEST") == 0){
   		input_dim = 32;
   		printf("TEST\n");
   		model = 3;
   	}
   	else{
   		printf("Invalid Neural network\n");
   		exit(-1);
   	}
   	int Batch = atoi(argv[2]);
   	int channels = atoi(argv[3]);
	levels = atoi(argv[4]);
	PC_scheme = atoi(argv[5]);
   	
	
	//test_aggregation(levels,channels);
	//printf("Batch size : %d\n", Batch);
   	//exit(-1);
   	struct convolutional_network net = init_network(model,Batch,channels);
	
	
		
	//check_dataset(Batch,  input_dim);
	   	clock_t start,end;
	   	
		net = feed_forward(X, net,channels);
	   	net = back_propagation(net);
	   	
		


		vector<F> witness;
	   	vector<F> new_model;
	   	vector<vector<F>> witness_matrix,model_matrix;
		vector<commitment> comm(2);
		
		proving_time = 0.0;
	
		
		get_witness(net,witness);
	   	get_model(net,new_model);
	   	witness.insert(witness.end(),new_model.begin(),new_model.end());
		pad_vector(witness);
	   	
		printf("Witness size : %d\n", witness.size());
		
		poly_commit(witness, witness_matrix, comm[0],levels);
	   	float commitment_time = proving_time;
		printf("Commit size : %d, Commit time : %lf\n",witness.size(),new_model.size(),proving_time);
		proving_time = 0.0;
		
		clock_t wc1,wc2;
		wc1 = clock();
		prove_feedforward(net);
		prove_backprop(net);
	   	wc2 = clock();
	   	proving_time = 0.0;
		
		float PoGDsize = proof_size(Transcript);
	   	vector<F> POGD_hashses = x_transcript;
		x_transcript.clear();
		check_dataset(channels*Batch,input_dim);
	   	POGD_hashses.insert(POGD_hashses.end(), x_transcript.begin(),x_transcript.end());
		x_transcript.clear();
		float data_prove_time = proving_time;
	   	proving_time = 0.0;
	   	
		net.Filters.clear();
		net.Filters = vector<vector<vector<vector<vector<F>>>>>();
		net.Rotated_Filters.clear();
		net.Rotated_Filters = vector<vector<vector<vector<vector<F>>>>>();
		net.Weights.clear();
		net.Weights = vector<vector<vector<F>>>();
		net.der.clear();
		net.der = vector<vector<vector<vector<vector<F>>>>>();
		//clear_witness(net);
		//clear_model(net);
		net.avg_backprop.clear();
		net.avg_backprop = vector<avg_layer_backprop>();
		net.avg_layers.clear();
		net.avg_layers = vector<avg_layer>();
		//net.convolutions.clear();
		//net.convolutions = vector<convolution_layer>();
		for(int i = 0; i < net.convolutions.size(); i++){
			net.convolutions[i].X.clear();
			net.convolutions[i].X = vector<vector<F>>();
			net.convolutions[i].fft_X.clear();
			net.convolutions[i].fft_X = vector<vector<F>>();
			net.convolutions[i].fft_W.clear();
			net.convolutions[i].fft_W = vector<vector<F>>();
			net.convolutions[i].Prod.clear();
			net.convolutions[i].Prod = vector<vector<F>>();
			net.convolutions[i].U.clear();
			net.convolutions[i].U = vector<vector<F>>();

			net.convolutions[i].Out.clear();
			net.convolutions[i].Out = vector<vector<F>>();
			net.convolutions[i].Sum.clear();
			net.convolutions[i].Sum = vector<vector<F>>();
			net.convolutions[i].Remainder.clear();
			net.convolutions[i].Remainder = vector<vector<F>>();	
		}
		
		//net.convolutions_backprop.clear();
		//net.convolutions_backprop = vector<convolution_layer_backprop>();
		for(int i = 0; i < net.convolutions_backprop.size(); i++){
			net.convolutions_backprop[i].der_prev.clear();
			net.convolutions_backprop[i].der_prev = vector<vector<vector<vector<F>>>>();
			net.convolutions_backprop[i].fft_X.clear();
			net.convolutions_backprop[i].fft_X = vector<vector<F>>();
			net.convolutions_backprop[i].derr.clear();
			net.convolutions_backprop[i].derr = vector<vector<F>>();
			net.convolutions_backprop[i].fft_der.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].fft_der = vector<vector<F>>();
			net.convolutions_backprop[i].Prod.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].Prod = vector<vector<F>>();
			net.convolutions_backprop[i].Rotated_W.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].Rotated_W = vector<vector<F>>();
			net.convolutions_backprop[i].pad_der.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].pad_der = vector<vector<F>>();
			net.convolutions_backprop[i].fft_pad_der.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].fft_pad_der = vector<vector<F>>();
			net.convolutions_backprop[i].rot_W.clear();//= vector<<vector<F>>();
			net.convolutions_backprop[i].rot_W = vector<vector<F>>();
			net.convolutions_backprop[i].fft_rot_W.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].fft_rot_W = vector<vector<F>>();
			net.convolutions_backprop[i].Prod_dx.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].Prod_dx = vector<vector<F>>();
			net.convolutions_backprop[i].U_dx.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].U_dx = vector<vector<F>>();
			net.convolutions_backprop[i].dx.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].dx = vector<vector<F>>();
			net.convolutions_backprop[i].U_dx_shifted.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].U_dx_shifted = vector<vector<F>>();
			net.convolutions_backprop[i].U_dx_remainders.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].U_dx_remainders = vector<vector<F>>();
			net.convolutions_backprop[i].dw.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].dw = vector<vector<F>>();
			net.convolutions_backprop[i].dw_remainders.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].dw_remainders = vector<vector<F>>();
			net.convolutions_backprop[i].Reduced_fft_dw.clear();// = vector<<vector<F>>();
			net.convolutions_backprop[i].Reduced_fft_dw = vector<vector<F>>();
			net.convolutions_backprop[i].padding.clear();// = vector<vector<vector<vector<F>>>>();
			net.convolutions_backprop[i].padding = vector<vector<vector<vector<F>>>>();
			
	}
		//net.relus.clear();
		//net.relus = vector<relu_layer>();
		for(int i = 0; i < net.relus.size(); i++){
			net.relus[i].input.clear();// = vector<F>();
			net.relus[i].input = vector<F>();
			net.relus[i].new_input.clear();// = vector<F>();
			net.relus[i].new_input = vector<F>();
			net.relus[i].output.clear();// = vector<F>();
			net.relus[i].output = vector<F>();
			net.relus[i].temp_output.clear();// = vector<F>();
			net.relus[i].temp_output = vector<F>();
			net.relus[i].most_significant_bits.clear();// = vector<F>();
			net.relus[i].most_significant_bits = vector<F>();
		}
		//net.fully_connected.clear();
		//net.fully_connected = vector<fully_connected_layer>();
		for(int i = 0; i < net.fully_connected.size(); i++){
			net.fully_connected[i].Z_new.clear();// = vector<vector<F>>();
			net.fully_connected[i].Z_new = vector<vector<F>>();
			net.fully_connected[i].Z_prev.clear();// = vector<vector<F>>();
			net.fully_connected[i].Z_prev = vector<vector<F>>();
			net.fully_connected[i].W.clear();// = vector<vector<F>>();
			net.fully_connected[i].W = vector<vector<F>>();
			net.fully_connected[i].Remainders.clear();// = vector<vector<F>>();
			net.fully_connected[i].Remainders = vector<vector<F>>();
			net.fully_connected[i].Z_temp.clear();// = vector<vector<F>>();
			net.fully_connected[i].Z_temp = vector<vector<F>>();
		}
		//net.fully_connected_backprop.clear();
		//net.fully_connected_backprop = vector<dense_layer_backprop>();
		for(int i = 0; i < net.fully_connected_backprop.size(); i++){
			net.fully_connected_backprop[i].dw.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].dw = vector<vector<F>>();
			net.fully_connected_backprop[i].dw_remainders.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].dw_remainders = vector<vector<F>>();
			net.fully_connected_backprop[i].Z.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].Z = vector<vector<F>>();
			net.fully_connected_backprop[i].W.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].W = vector<vector<F>>();
			net.fully_connected_backprop[i].dx_input.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].dx_input = vector<vector<F>>();
			net.fully_connected_backprop[i].dx_remainders.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].dx_remainders = vector<vector<F>>();
			net.fully_connected_backprop[i].dw_remainders.clear();// = vector<vector<F>>();
			net.fully_connected_backprop[i].dw_remainders = vector<vector<F>>();
		}
		net.relus_backprop.clear();
		net.relus_backprop = vector<relu_layer_backprop>();
		
		
		
		witness.clear();
		//witness = vector<F>();
		vector<F>().swap(witness);
		
		// Start aggregation. Because a complete IVC version is not ready, 
		// we use the same witness/model matrixes 
		
		
		vector<vector<vector<vector<F>>>> witness_matrixes(1);
		
		witness_matrixes[0].push_back(witness_matrix);
		witness_matrixes[0].push_back(witness_matrix);
		witness_matrix.clear();
		
		vector<vector<commitment>> comm_aggr(1);
		comm_aggr[0].push_back(comm[0]);
		comm_aggr[0].push_back(comm[0]);
		
		vector<vector<__hhash_digest>>().swap(comm[0].hashes_sha);

		vector<F> proof_witness = prove_aggr(witness_matrixes,comm_aggr); 
		vector<vector<__hhash_digest>>().swap(comm_aggr[0][0].hashes_sha);
		vector<vector<__hhash_digest>>().swap(comm_aggr[0][1].hashes_sha);
		
		proof_witness.insert(proof_witness.end(),POGD_hashses.begin(),POGD_hashses.end());
		//proof_witness.push_back()

		float aggregation_time_recursion = proving_time;
		//printf("Aggregation time : %lf \n", aggregation_time);
		//printf("Aggregation time V circuit : %lf \n", aggregation_time_recursion);
		
		proving_time = 0.0;
	   	
		printf("PoGD Proving time : %lf,%lf,%lf,%lf\n",Forward_propagation,dx_computation,gradients_computation,Forward_propagation + dx_computation + gradients_computation );
		printf("Checking/Updating Input : %lf\n",data_prove_time);
	   
		
		
	   	printf("PoGD proof size / PoGD dataset check size : %lf, %lf\n",PoGDsize,proof_size(Transcript) );
	   	
		
	   //printf("Aggregation time  : %lf, %d\n",aggregation_time, proof_witness.size());
	   	
		
		
		
		vector<struct proof> u_proof_temp,u_proof;
	   	
		proving_time = 0.0;
   		
   		//Proof of proof of aggregation verifier circuit
   		
   		
		
	   	// Use a temporary proof to generate the actual proof_u (the proof of the verifier from previous step)
	   	vector<F> aggregation_hashes = x_transcript;
		x_transcript.clear();
		
		u_proof_temp.push_back(verify_proof(Transcript));
	   	vector<proof> temp_P = mimc_sumcheck(POGD_hashses);
		u_proof_temp.insert(u_proof_temp.end(),temp_P.begin(),temp_P.end());
		temp_P = mimc_sumcheck(aggregation_hashes);
		u_proof_temp.insert(u_proof_temp.end(),temp_P.begin(),temp_P.end());
		proof_witness.insert(proof_witness.end(),x_transcript.begin(),x_transcript.end());
   		vector<F> proof_hashes = x_transcript;
		pad_vector(proof_witness);
		vector<vector<F>> proof_witness_matrix;
		commitment proof_com;
		printf("proof witness size : %d\n",proof_witness.size());
		proving_time =0.0;
		
		
		
		poly_commit(proof_witness, proof_witness_matrix, proof_com,levels);
		
		commitment_time += proving_time;
		vector<vector<vector<vector<F>>>> proof_witness_matrixes(1);
		vector<vector<commitment>> proof_comm_aggr(1);proof_comm_aggr[0].push_back(proof_com);proof_comm_aggr[0].push_back(proof_com);
		proof_witness_matrixes[0].push_back(proof_witness_matrix);
		proof_witness_matrixes[0].push_back(proof_witness_matrix);
		
		
		proving_time = 0.0;
		x_transcript.clear();
		prove_aggr(proof_witness_matrixes,proof_comm_aggr);
		
		aggregation_time_recursion += proving_time;
		//prove_aggr(witness_matrixes,comm_aggr);
		aggregation_hashes.insert(aggregation_hashes.begin(),x_transcript.begin(),x_transcript.end());
		proof_hashes.insert(proof_hashes.end(),aggregation_hashes.begin(),aggregation_hashes.end());
		// Get the U and also append the PoGD proof 
   		u_proof.push_back(verify_proof(u_proof_temp));
   		temp_P = mimc_sumcheck(proof_hashes);
		u_proof.insert(u_proof.end(),temp_P.begin(),temp_P.end());
		//u_proof.push_back(prove_aggr());
   		u_proof.insert(u_proof.end(),Transcript.begin(),Transcript.end());
		
		//u_proof.push_back(Transcript);
   		// Runs the NAGG verifiers
   	   	//u_proof.push_back(prove_aggr());
   	   	// Get the transcript size
		
		
		
		proving_time = 0.0;
	   
	   	vector<proof> pr;
   	   	//pr.push_back(verify_proof(u_proof));
   	   	vector<F> total_input;
		proof_hashes.insert(proof_hashes.end(),POGD_hashses.begin(),POGD_hashses.end());
		
		mimc_sumcheck(proof_hashes);
		//pr.push_back(verify_hashes(u_proof));
   	   	//printf("%d\n",get_transcript_size(pr));
		printf("Verification time : %lf\n", verify(u_proof));
   		printf("Proving verifier circuit : %lf\n",proving_time );
		printf("Proving aggregation recursion circuit : %lf\n",aggregation_time_recursion );
		printf("Recursion P : %lf\n",aggregation_time_recursion+proving_time );
		printf("Aggregation time : %lf\n", aggregation_time);
	   	printf("Commitment time : %lf\n", commitment_time);
		printf("Final proof size : %lf\n",proof_size(u_proof));
		//printf("Witness size (bits) : %d, Commit PoGD : %d,  Commit Verifier Circuit :  %d \n",sizes[0],sizes[1]+sizes[2]+partitions*Batch*input_dim*input_dim,sizes[3] );


	return 0;
}