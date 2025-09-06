# Expected Results for RQ2

## Overview

RQ2 tests the model's ability to generalize to unseen screen resolutions using leave-one-out evaluation across 9 different resolutions.

## Expected Performance per Resolution

Each leave-one-out cycle should achieve:

- **Detection Rate @ 1% FPR**: ≥ 99%
- **AUROC Score**: ≥ 0.999

## Directory to Resolution Mapping

For reviewer clarity, the following mapping exists between directory names and actual screen resolutions:

**Landscape Resolutions:**

- `1.land_excluded` → 1366x768 (Standard laptop resolution)
- `2.land_excluded` → 1920x998 (Full HD desktop with browser chrome)
- `3.land_excluded` → 1478x837 (Mid-range laptop resolution)
- `4.land_excluded` → 1536x824 (High-DPI laptop resolution)
- `5.land_excluded` → 1366x728 (Standard laptop with reduced height)

**Portrait Resolutions:**

- `1.port_excluded` → 800x1280 (Tablet portrait mode)
- `2.port_excluded` → 414x896 (iPhone X/11/12 series)
- `3.port_excluded` → 768x1024 (iPad portrait mode)
- `4.port_excluded` → 360x640 (Android phone standard)

## Specific Resolution Results

Based on Table 3 in the paper:

| Directory Name  | Test Resolution | Expected DR@1%FPR | Expected AUROC |
| --------------- | --------------- | ----------------- | -------------- |
| 1.land_excluded | 1366x768        | 1.0               | 1.0            |
| 1.port_excluded | 800x1280        | 1.0               | 1.0            |
| 2.land_excluded | 1920x998        | 1.0               | 1.0            |
| 2.port_excluded | 414x896         | 1.0               | 1.0            |
| 3.land_excluded | 1478x837        | 1.0               | 0.999          |
| 3.port_excluded | 768x1024        | 1.0               | 1.0            |
| 4.land_excluded | 1536x824        | 1.0               | 1.0            |
| 4.port_excluded | 360x640         | 0.990             | 0.999          |
| 5.land_excluded | 1366x728        | 1.0               | 1.0            |

## Global Performance

- **Overall Detection Rate @ 1% FPR**: 99.8%
- **Overall AUROC**: 0.999

## Output Structure

```
./out/l1o_res/
├── 1land_excluded/            # 1366x768 resolution held out
│   ├── ep_X/                  # Best epoch results
│   │   └── eval_1/
│   │       ├── eval_metrics.txt
│   │       ├── roc_curve.png
│   │       ├── y_true.npy
│   │       ├── y_scores.npy
│   │       └── vis_conf/
├── 1.port_excluded/            # 800x1280 resolution held out
│   └── ...
├── 2.land_excluded/            # 1920x998 resolution held out
│   └── ...
├── 2.port_excluded/            # 414x896 resolution held out
│   └── ...
├── 3.land_excluded/            # 1478x837 resolution held out
│   └── ...
├── 3.port_excluded/            # 768x1024 resolution held out
│   └── ...
├── 4.land_excluded/            # 1536x824 resolution held out
│   └── ...
├── 4.port_excluded/            # 360x640 resolution held out
│   └── ...
└── 5.land_excluded/            # 1366x728 resolution held out
    └── ...
```

## Key Metrics to Verify

1. Each **eval_metrics.txt** should show DR@1%FPR ≥ 0.99 and AUC ≥ 0.999
2. **ROC curves** should demonstrate strong performance across all resolutions
3. **Confusion matrices** should show consistent high accuracy regardless of resolution
