#!/bin/bash

# RQ1 & RQ4: Can PP_det accurately identify new instances of BMAs and fresh BMA attacks?
# This script runs the evaluation for Research Questions 1 and 4

echo "Running RQ1 & RQ4 evaluation..."
echo "Testing model's ability to identify new BMA instances and fresh attacks"

# Navigate to the project root, then to the train_test directory
cd ../../artifacts/train_test

# Run the combined evaluation for RQ1 and RQ4
python tt_comb.py --use_pt_model True

echo "RQ1 & RQ4 evaluation complete!"
echo "Results saved to ./out/comb/"
echo "Check the following for metrics:"
echo "- ./out/comb/ep_1/eval_1/ (RQ1 results)"
echo "- ./out/comb/ep_1/eval_2/ (RQ4 results)"
