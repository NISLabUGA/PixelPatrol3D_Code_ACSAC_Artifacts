# Expected Results for Training Verification: RQ2 Resolution Models

## Overview

This resolution training verification demonstrates that the leave-one-out resolution training process for RQ2 works correctly and shows how different epoch models were selected for different resolutions.

## Expected Training Behavior

### Training Time

- **Per Epoch**: ~5 minutes (V100 32GB GPU)
- **Verification (1-2 epochs, subset of resolutions)**: ~15-30 minutes (recommended for reviewers)
- **Full Training (9 resolutions, 10 epochs max)**: ~7.5 hours total (not necessary for verification)

### Leave-One-Out Training Progress

The verification run should demonstrate:

1. **Leave-One-Out Cycles**: Each resolution held out while training on others
2. **Resolution-Agnostic Processing**: Scaling and padding handles different dimensions
3. **Cross-Resolution Evaluation**: Performance tested on held-out resolutions
4. **Model Checkpointing**: Separate models saved for each resolution split

### Expected Output Structure

```
artifacts/train_test/out/l1o_res/
├── training_log.log                    # Overall training progress
├── notes.txt                          # Resolution training configuration
├── 1land_excluded/                     # 1366x768 held out
│   ├── res_detector_1_epoch_1.pth     # Epoch 1 checkpoint
│   ├── res_detector_1_epoch_2.pth     # Epoch 2 checkpoint
│   ├── res_detector_1_epoch_3.pth     # Epoch 3 checkpoint
│   └── ep_3/                          # Final epoch evaluation
│       └── eval_1/                    # Held-out resolution results
│           ├── eval_metrics.txt
│           ├── roc_curve.png
│           ├── y_true.npy
│           ├── y_scores.npy
│           └── vis_conf/
├── 2port_excluded/                     # 414x896 held out
│   ├── res_detector_2_epoch_1.pth
│   ├── res_detector_2_epoch_2.pth
│   ├── res_detector_2_epoch_3.pth
│   └── ep_3/
│       └── eval_1/
└── 3land_excluded/                     # 1478x837 held out
    ├── res_detector_3_epoch_1.pth
    ├── res_detector_3_epoch_2.pth
    ├── res_detector_3_epoch_3.pth
    └── ep_3/
        └── eval_1/
```

## Key Metrics to Verify

### Training Log Verification

Check `training_log.log` for:

- Successful resolution preprocessing for different dimensions
- Consistent training across different resolution combinations
- Cross-resolution validation performance
- Resolution-agnostic feature learning progress

### Resolution Generalization

Expected patterns:

- **Landscape Resolutions**: Similar performance patterns across different widths
- **Portrait Resolutions**: Consistent performance despite aspect ratio differences
- **Size Independence**: Performance maintained from small (414x896) to large (1920x998)
- **Aspect Ratio Robustness**: Good generalization between landscape and portrait

## Understanding Epoch Selection

### Why Different Epochs for Different Resolutions

The verification demonstrates early training, but in full training:

- **Simple Resolutions**: May converge quickly (epochs 5-7)
- **Complex Resolutions**: May require more training (epochs 8-10)
- **Aspect Ratio Impact**: Portrait vs. landscape may affect convergence
- **Size Complexity**: Larger resolutions may need more epochs

### Resolution Training Dynamics

The verification should show:

- **Preprocessing Effectiveness**: Scaling and padding handle arbitrary dimensions
- **Feature Generalization**: Visual features become resolution-agnostic
- **Consistent Learning**: Similar training patterns across resolutions

## Resolution Mapping Reference

### Verification Subset Resolutions

- **1land_excluded** → 1366x768 (Standard laptop resolution)
- **2port_excluded** → 414x896 (iPhone X/11/12 series)
- **3land_excluded** → 1478x837 (Mid-range laptop resolution)

### Full Training Resolution Models

**Landscape Resolutions:**

- `m_res_1_land_ep10.pth` → 1366x768 (epoch 10)
- `m_res_2_land_ep7.pth` → 1920x998 (epoch 7)
- `m_res_3_land_ep8.pth` → 1478x837 (epoch 8)
- `m_res_4_land_ep10.pth` → 1536x824 (epoch 10)
- `m_res_5_land_ep9.pth` → 1366x728 (epoch 9)

**Portrait Resolutions:**

- `m_res_1_port_ep9.pth` → 800x1280 (epoch 9)
- `m_res_2_port_ep7.pth` → 414x896 (epoch 7)
- `m_res_3_port_ep5.pth` → 768x1024 (epoch 5)
- `m_res_4_port_ep8.pth` → 360x640 (epoch 8)

## Troubleshooting

### Common Issues

1. **Memory Issues**: Different resolutions may have varying memory requirements
2. **Preprocessing Errors**: Verify scaling and padding pipeline works correctly
3. **Poor Generalization**: Check that resolution-agnostic features are learned
4. **Convergence Variation**: Different resolutions may converge at different rates

### Expected Warnings

Normal warnings that may appear:

- Resolution preprocessing notifications
- Aspect ratio handling messages
- Memory allocation adjustments
- Cross-resolution evaluation progress

## Validation Criteria

The resolution training verification is successful if:

1. **Training Completes**: All resolution cycles finish without errors
2. **Consistent Generalization**: Similar performance patterns across resolutions
3. **Files Generated**: All expected output files for each resolution split
4. **Resolution Independence**: Good performance regardless of aspect ratio or size

## Verification vs. Full Training

### Recommended Verification (1-2 epochs, subset of resolutions)

- **Purpose**: Verify leave-one-out resolution training pipeline works correctly
- **Time**: ~15-30 minutes
- **Sufficient to demonstrate**: Resolution preprocessing, cross-resolution evaluation, model checkpointing
- **Cost-effective**: Minimal compute resources required

### Optional Full Training (9 resolutions, 10 epochs max)

- **Purpose**: Reproduce exact resolution model performance
- **Time**: ~7.5 hours total
- **Not necessary for verification**: Claims 1-4 already provide reproducibility testing
- **Individual Convergence**: Each resolution typically converges by epoch 5-10
- **Global Performance**: >95% detection rate across all held-out resolutions
- **Resolution Robustness**: Consistent performance from 360x640 to 1920x998

**Note**: Running 1-2 epochs on a subset of resolutions is sufficient to verify the resolution training process works correctly and is the recommended approach for reviewers. Full training is not necessary since claims 1-4 already handle reproducibility verification.

## Key Insights

This verification demonstrates:

- **Resolution Agnosticism**: Models work across diverse screen sizes
- **Aspect Ratio Independence**: Performance maintained for landscape and portrait
- **Preprocessing Effectiveness**: Scaling and padding enable arbitrary dimensions
- **Feature Robustness**: Visual features generalize across pixel densities

## Expected Convergence Patterns

### Fast Converging Resolutions

- **iPad Portrait (768x1024)**: Often converges by epoch 5
- **iPhone Series (414x896)**: Typically converges by epoch 7

### Slower Converging Resolutions

- **Standard Laptop (1366x768)**: May require epoch 10
- **High-DPI Laptop (1536x824)**: Often needs epoch 10

The resolution training verification validates that the resolution-agnostic approach successfully enables generalization to never-before-seen screen resolutions, supporting the RQ2 claims about cross-resolution detection capabilities.
