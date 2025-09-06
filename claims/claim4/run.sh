#!/bin/bash

# RQ5: Can PP_det be strengthened against adversarial examples?
# This script runs both before and after adversarial training evaluations

echo "Running RQ5 evaluation..."
echo "Testing model robustness against adversarial examples"

# Navigate to the project root, then to the train_test directory
cd ../../artifacts/train_test

echo "=== Running evaluation BEFORE adversarial training ==="
python tt_comb_adv.py --use_pt_model True --pt_model_path ../models/rq5/m_no_adv_ep4.pth

echo ""
echo "=== Running evaluation AFTER adversarial training ==="
python tt_comb_adv.py --use_pt_model True --pt_model_path ../models/rq5/m_adv_ep7.pth

echo ""
echo "RQ5 evaluation complete!"
echo "Results saved to:"
echo "- ./out/comb_no_adv/ (before adversarial training)"
echo "- ./out/comb_adv/ (after adversarial training)"
echo ""
echo "Compare the following metrics across adversarial levels:"
echo "- eval_0/ (clean examples)"
echo "- eval_1/ (level 1 adversarial)"
echo "- eval_2/ (level 2 adversarial)"
echo "- eval_3/ (level 3 adversarial)"
echo "- eval_4/ (level 4 adversarial)"
echo "- eval_5/ (level 5 adversarial)"
