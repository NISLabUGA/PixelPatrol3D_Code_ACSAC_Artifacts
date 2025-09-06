# Expected Results for RQ1 & RQ4

## RQ1 Results (eval_1)

- **Detection Rate @ 1% FPR**: Above 99%
- **AUROC Score**: 1.0
- **Test Set**: 500 new BMA instances from previously seen campaigns + 500 benign samples

## RQ4 Results (eval_2)

- **Detection Rate @ 1% FPR**: Above 97%
- **AUROC Score**: Above 0.999
- **Test Set**: 138 fresh BMA samples collected months after training + 500 benign samples

## Output Structure

```
./out/comb/ep_1/
├── eval_1/                    # RQ1 results
│   ├── eval_metrics.txt       # Accuracy, precision, recall, F1, AUC, DR@1%FPR
│   ├── roc_curve.png         # ROC curve visualization
│   ├── y_true.npy            # Ground truth labels
│   ├── y_scores.npy          # Prediction scores
│   └── vis_conf/             # Visual confusion examples
│       ├── fp/               # False positives
│       ├── fn/               # False negatives
│       ├── tp/               # True positives (if enabled)
│       └── tn/               # True negatives (if enabled)
└── eval_2/                    # RQ4 results (same structure as eval_1)
```

## Key Metrics to Verify

1. **eval_metrics.txt** should show:

   - RQ1: DR@1%FPR ≥ 0.99, AUC = 1.0
   - RQ4: DR@1%FPR ≥ 0.97, AUC ≥ 0.999

2. **ROC curves** should demonstrate excellent separation between classes

3. **Confusion matrices** should show high true positive and true negative rates
