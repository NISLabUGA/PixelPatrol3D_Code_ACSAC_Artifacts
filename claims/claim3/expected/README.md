# Expected Results for RQ3

## Overview

RQ3 tests the model's ability to generalize to never-before-seen BMA campaigns using leave-one-out evaluation across 10 different campaigns.

## Expected Performance per Campaign

Each leave-one-out cycle should achieve:

- **Detection Rate @ 1% FPR**: ≥ 99%
- **AUROC Score**: ≥ 0.996

## Specific Campaign Results

Based on Table 4 in the paper:

| Campaign    | # Benign | # BMA | Expected AUROC | Expected DR@1%FPR |
| ----------- | -------- | ----- | -------------- | ----------------- |
| Campaign 1  | 500      | 281   | 1.0            | 0.993             |
| Campaign 2  | 500      | 229   | 1.0            | 1.0               |
| Campaign 3  | 500      | 59    | 0.998          | 1.0               |
| Campaign 4  | 500      | 67    | 1.0            | 1.0               |
| Campaign 5  | 500      | 910   | 0.999          | 0.998             |
| Campaign 6  | 500      | 15    | 0.996          | 1.0               |
| Campaign 7  | 500      | 43    | 1.0            | 1.0               |
| Campaign 8  | 500      | 83    | 1.0            | 1.0               |
| Campaign 9  | 500      | 224   | 1.0            | 1.0               |
| Campaign 10 | 500      | 128   | 1.0            | 1.0               |

## Global Performance

- **Overall Detection Rate @ 1% FPR**: 99.3%
- **Overall AUROC**: 0.999

## Campaign Categories

The 10 campaigns represent diverse BMA types:

- **Notification Stealing**: Multiple campaigns with fake video players
- **Fake Software Download**: Various software download scams
- **Service Sign-up Scam**: Registration-based scams
- **Fake Lottery/Sweepstakes**: Prize-based deception
- **Technical Support Scam**: Fake tech support warnings

## Output Structure

```
./out/l1o_camp/
├── 1.mclus_excluded/          # First campaign held out
│   ├── ep_X/                  # Best epoch results
│   │   └── eval_1/
│   │       ├── eval_metrics.txt
│   │       ├── roc_curve.png
│   │       ├── y_true.npy
│   │       ├── y_scores.npy
│   │       └── vis_conf/
├── 2.mclus_excluded/          # Second campaign held out
│   └── ...
├── ...
└── 10.mclus_excluded/         # Tenth campaign held out
    └── ...
```

## Key Metrics to Verify

1. Each **eval_metrics.txt** should show DR@1%FPR ≥ 0.99 and AUC ≥ 0.996
2. **ROC curves** should demonstrate strong performance across all campaign types
3. **Confusion matrices** should show consistent detection regardless of campaign visual style
4. **vis_conf/** folders should contain misclassified examples for analysis
