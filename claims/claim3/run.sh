#!/bin/bash

# RQ3: Can PP_det identify web pages belonging to never-before-seen BMA campaigns?
# This script runs the leave-one-out campaign evaluation for Research Question 3

echo "Running RQ3 evaluation..."
echo "Testing model's ability to generalize to unseen BMA campaigns"

# Navigate to the project root, then to the train_test directory
cd ../../artifacts/train_test

# Run the leave-one-out campaign evaluation for RQ3
python3 tt_l1o_camp.py --use_pt_model True

echo "RQ3 evaluation complete!"
echo "Results saved to ./out/l1o_camp/"
echo "Check the following directories for metrics from each leave-one-out cycle:"
echo "- ./out/l1o_camp/1.mclus_excluded/"
echo "- ./out/l1o_camp/2.mclus_excluded/"
echo "- ./out/l1o_camp/3.mclus_excluded/"
echo "- ./out/l1o_camp/4.mclus_excluded/"
echo "- ./out/l1o_camp/5.mclus_excluded/"
echo "- ./out/l1o_camp/6.mclus_excluded/"
echo "- ./out/l1o_camp/7.mclus_excluded/"
echo "- ./out/l1o_camp/8.mclus_excluded/"
echo "- ./out/l1o_camp/9.mclus_excluded/"
echo "- ./out/l1o_camp/10.mclus_excluded/"
