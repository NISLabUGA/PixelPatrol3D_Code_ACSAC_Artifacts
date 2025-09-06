# Expected Results for RQ2

## Overview

RQ2 tests the model's ability to generalize to unseen screen resolutions using leave-one-out evaluation across 9 different resolutions.

## Expected Performance per Resolution

Each leave-one-out cycle should achieve:

- **Detection Rate @ 1% FPR**: ≥ 99%
- **AUROC Score**: ≥ 0.999

## Specific Resolution Results

Based on Table 3 in the paper:

| Test Resolution | Expected DR@1%FPR | Expected AUROC |
| --------------- | ----------------- | -------------- |
| 1366x768        | 1.0               | 1.0            |
| 800x1280        | 1.0               | 1.0            |
| 1920x998        | 1.0               | 1.0            |
| 414x896         | 1.0               | 1.0            |
| 1478x837        | 1.0               | 0.999          |
| 768x1024        | 1.0               | 1.0            |
| 1536x824        | 1.0               | 1.0            |
| 360x640         | 0.990             | 0.999          |
| 1366x728        | 1.0               | 1.0            |

## Global Performance

- **Overall Detection Rate @ 1% FPR**: 99.8%
- **Overall AUROC**: 0.999

## Output Structure

```
./out/l1o_res/
├── 1.mclus_excluded/          # First resolution held out
│   ├── ep_X/                  # Best epoch results
│   │   └── eval_1/
│   │       ├── eval_metrics.txt
│   │       ├── roc_curve.png
│   │       ├── y_true.npy
│   │       ├── y_scores.npy
│   │       └── vis_conf/
├── 2.mclus_excluded/          # Second resolution held out
│   └── ...
├── ...
└── 9.mclus_excluded/          # Ninth resolution held out
    └── ...
```

## Key Metrics to Verify

1. Each **eval_metrics.txt** should show DR@1%FPR ≥ 0.99 and AUC ≥ 0.999
2. **ROC curves** should demonstrate strong performance across all resolutions
3. **Confusion matrices** should show consistent high accuracy regardless of resolution
