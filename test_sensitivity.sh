#!/bin/bash

cd /Users/liuhengyu/Desktop/kaizen
chmod +x build.sh
./build.sh

if [ $? -ne 0 ]; then
    echo "Build failed! Please check compilation errors."
    exit 1
fi


echo "Test 1: LeNet sensitivity analysis for digit 7"
echo "-----"
./main LENET 1 1 1 2 7
echo ""
