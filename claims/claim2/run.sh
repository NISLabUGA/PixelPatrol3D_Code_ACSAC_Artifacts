#!/bin/bash

# RQ2: Can PP_det accurately identify instances of BMAs captured on new screen sizes?
# This script runs the leave-one-out resolution evaluation for Research Question 2

echo "Running RQ2 evaluation..."
echo "Testing model's ability to generalize to unseen screen resolutions"

# Navigate to the project root, then to the train_test directory
cd ../../artifacts/train_test

# Run the leave-one-out resolution evaluation for RQ2
python tt_l1o_res.py --use_pt_model True

echo "RQ2 evaluation complete!"
echo "Results saved to ./out/l1o_res/"
echo "Check the following directories for metrics from each leave-one-out cycle:"
echo "- ./out/l1o_res/1.mclus_excluded/"
echo "- ./out/l1o_res/2.mclus_excluded/"
echo "- ./out/l1o_res/3.mclus_excluded/"
echo "- ./out/l1o_res/4.mclus_excluded/"
echo "- ./out/l1o_res/5.mclus_excluded/"
echo "- ./out/l1o_res/6.mclus_excluded/"
echo "- ./out/l1o_res/7.mclus_excluded/"
echo "- ./out/l1o_res/8.mclus_excluded/"
echo "- ./out/l1o_res/9.mclus_excluded/"
