# Model Sensitivity Zero-Knowledge Proof

This implementation extends the existing ZKP framework for DNN training to provide **zero-knowledge proofs of model sensitivity analysis**. Given a trained model and input, it proves the correctness of gradient computation from output to input features while maintaining model privacy.

## Key Features

### 🔬 **Sensitivity Analysis**
- **Public Input**: Uses randomly generated or specified input X (made public)
- **Private Model**: Model weights remain completely confidential throughout the proof
- Computes gradients ∂output/∂input_features using backpropagation on public input
- Quantifies input feature importance for model predictions
- Supports different target output classes for analysis

### 🔐 **Zero-Knowledge Properties** 
- **Soundness**: Invalid gradients cannot produce valid proofs via GKR soundness
- **Completeness**: Correct gradients always verify through existing proof system
- **Zero-Knowledge**: Model weights remain private through commitment schemes
- **Verifiability**: Gradients proven correct without revealing intermediate computations

### 🏗️ **Architecture Integration**
- Uses existing GKR infrastructure (`prove_dx_prod`, `prove_dot_x_prod`)
- Integrates with polynomial commitment schemes for aggregation  
- Follows same proving patterns as DNN training ZKP in `main.cpp`
- Supports CNN (LeNet, AlexNet, VGG) and dense layer architectures

## Implementation Details

### Core Function: `prove_model_sensitivity()`

Located in `src/main.cpp:1832-1940`, this function:

1. **Extracts Input Gradients**: Gets dx from first layer backpropagation
2. **Layer-wise Proof Generation**: Proves gradient flow through each layer
   - Convolution layers: Uses `prove_dx_prod()` for gradient correctness
   - Dense layers: Uses `prove_dot_x_prod()` for matrix-vector gradients
3. **Sensitivity Quantification**: Computes L2 norm of input gradients
4. **Gradient Commitment**: Uses polynomial commitments to bind all proofs
5. **Transcript Accumulation**: Adds proofs to global transcript for verification

### Protocol Flow

```
Public Input X + Private Model + Target Output
          ↓
    Forward Pass (inference on public X)
          ↓  
    Backward Pass (compute ∂output/∂X)
          ↓
    Layer-wise Gradient Proofs (GKR)
          ↓
    Aggregate with Poly Commitments  
          ↓
    Sensitivity Score + ZKP
```

### Integration Points

- **Main Function** (`main.cpp:2617-2623`): Calls sensitivity analysis after standard training proofs
- **Gradient Extraction**: Uses existing `convert2vector()` to flatten gradient tensors
- **Proof System**: Reuses `Transcript` vector and `predicates_size` tracking
- **Randomness**: Uses same `generate_randomness()` and `evaluate_vector()` functions

## Usage

### Command Line
```bash
# Basic sensitivity analysis for digit 7
./main LENET 1 1 1 2 7

# Sensitivity for different target classes
./main LENET 1 1 1 2 3    # digit 3
./main TEST 1 1 1 2 0     # class 0

# Different model architectures  
./main AlexNet 1 1 1 2 5  # AlexNet for class 5
./main VGG 1 1 1 2 9      # VGG for class 9
```

### Parameters
- `MODEL`: Neural network architecture (LENET, AlexNet, VGG, TEST)
- `BATCH`: Batch size (typically 1 for sensitivity analysis)
- `CHANNELS`: Input channels 
- `LEVELS`: Polynomial commitment levels
- `PC_SCHEME`: Polynomial commitment scheme
- `TARGET_CLASS`: Output class for sensitivity analysis (0-9)

### Test Script
```bash
chmod +x test_sensitivity.sh
./test_sensitivity.sh
```

## Output Analysis

The system provides detailed output including:

- **Input Gradient Size**: Number of input features analyzed
- **Layer-wise Proofs**: Verification of gradient computation through each layer
- **Sensitivity Score**: L2 norm quantifying input feature importance
- **ZKP Statistics**: Number of proofs generated and transcript size
- **Commitment Info**: Polynomial commitment verification

### Example Output
```
=== Starting Model Sensitivity ZKP ===
Public input dimensions: 1x1x32x32
PRIVACY NOTE: Model weights remain confidential throughout the proof
Only gradients w.r.t. public input will be proven correct
Setting up gradient computation for target output: 7
Running backward pass for sensitivity analysis...
Input gradients size: 1024, evaluation: 0x...
Proving gradient flow from output to input features...
✓ Layer 2 sensitivity gradient proven
✓ Layer 1 sensitivity gradient proven  
✓ Dense layer 0 sensitivity gradient proven
Final sensitivity magnitude: 0x...
✓ Input sensitivity gradients committed
✓ Sensitivity ZKP construction complete
✓ Total gradient proofs: 5

=== Model Sensitivity Analysis Results ===
Target output class: 7
Input features analyzed: 1024
Sensitivity score (L2 norm): 0x...
ZKP proofs generated: 5

=== Privacy Guarantees ===
✓ Public input X is known to all parties
✓ Model weights remain completely private
✓ Gradients ∂output/∂input are cryptographically proven correct
✓ No information about model parameters leaked in proofs
```

## Applications

### 🎯 **Explainable AI**
- Prove which input features most influence model decisions
- Verify gradient-based attribution methods (e.g., saliency maps)
- Audit model behavior for specific classes

### 🛡️ **Model Auditing** 
- Verify model sensitivity without revealing weights
- Prove compliance with fairness constraints on feature importance
- Audit for adversarial robustness via gradient analysis

### 🔍 **Debugging & Validation**
- Verify backpropagation implementation correctness
- Prove gradient computation in privacy-preserving ML
- Validate feature importance rankings

## Technical Advantages

1. **Efficiency**: Reuses existing GKR circuits and commitment schemes
2. **Modularity**: Integrates seamlessly with existing ZKP infrastructure  
3. **Generality**: Supports multiple architectures without modification
4. **Privacy**: Model weights never revealed during sensitivity analysis
5. **Verifiability**: Cryptographically guaranteed gradient correctness

## Future Extensions

- **Multi-output Sensitivity**: Analyze gradients for multiple output classes simultaneously
- **Higher-order Gradients**: Extend to Hessian-based sensitivity analysis  
- **Differential Privacy**: Add noise to gradients while maintaining proof validity
- **Batch Processing**: Optimize for multiple input samples with shared proofs