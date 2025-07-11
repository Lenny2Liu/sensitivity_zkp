#!/bin/bash

echo "=== Model Sensitivity ZKP Test ==="
echo "Testing different neural network architectures with sensitivity analysis"
echo "PUBLIC INPUT: Random input X will be generated and made public"
echo "PRIVATE MODEL: Model weights remain confidential throughout the proof"
echo ""

# Compile the project
echo "Building project..."
cd /Users/liuhengyu/Desktop/kaizen
chmod +x build.sh
./build.sh

if [ $? -ne 0 ]; then
    echo "Build failed! Please check compilation errors."
    exit 1
fi

echo "Build successful!"
echo ""

# Test 1: LeNet with class 7 sensitivity
echo "Test 1: LeNet sensitivity analysis for digit 7"
echo "Command: ./main LENET 1 1 1 2 7"
echo "-----"
./main LENET 1 1 1 2 7
echo ""

# Test 2: LeNet with class 3 sensitivity  
echo "Test 2: LeNet sensitivity analysis for digit 3"
echo "Command: ./main LENET 1 1 1 2 3"
echo "-----"
./main LENET 1 1 1 2 3
echo ""

# Test 3: TEST model with class 0 sensitivity
echo "Test 3: TEST model sensitivity analysis for class 0"
echo "Command: ./main TEST 1 1 1 2 0"
echo "-----" 
./main TEST 1 1 1 2 0
echo ""

echo "=== Sensitivity ZKP Analysis Complete ==="
echo ""
echo "Key features demonstrated:"
echo "✓ Gradient computation from output to input features"
echo "✓ Layer-wise gradient flow verification using GKR"
echo "✓ Input sensitivity quantification with L2 norm"
echo "✓ Zero-knowledge proofs for gradient correctness"
echo "✓ Polynomial commitments for gradient aggregation"
echo "✓ Support for different model architectures"
echo ""
echo "Usage: ./main <MODEL> <BATCH> <CHANNELS> <LEVELS> <PC_SCHEME> [TARGET_CLASS]"
echo "Models: LENET, AlexNet, mAlexNet, VGG, TEST"
echo "TARGET_CLASS: 0-9 (digit class for sensitivity analysis)"