# Expected Results for Training Verification: RQ3 Campaign Models

## Overview

This campaign training verification demonstrates that the leave-one-out campaign training process for RQ3 works correctly and shows how different epoch models were selected for different campaigns.

## Expected Training Behavior

### Leave-One-Out Campaign Training Progress

The verification run should demonstrate:

1. **Campaign Subset Selection**: 2-3 campaigns chosen for verification
2. **Leave-One-Out Cycles**: Each campaign held out while training on others
3. **Campaign-Agnostic Processing**: Features generalize across attack patterns
4. **Cross-Campaign Evaluation**: Performance tested on held-out campaigns
5. **Model Checkpointing**: Separate models saved for each campaign split

### Expected Output Structure

```
artifacts/train_test/out/l1o_camp/
├── training_log.log                    # Overall training progress
├── notes.txt                          # Campaign training configuration
├── camp_1_excluded/                    # Campaign 1 held out
│   ├── camp_detector_1_epoch_1.pth    # Epoch 1 checkpoint
│   ├── camp_detector_1_epoch_2.pth    # Epoch 2 checkpoint
│   ├── camp_detector_1_epoch_3.pth    # Epoch 3 checkpoint
│   └── ep_3/                          # Final epoch evaluation
│       └── eval_1/                    # Held-out campaign results
│           ├── eval_metrics.txt
│           ├── roc_curve.png
│           ├── y_true.npy
│           ├── y_scores.npy
│           └── vis_conf/
├── camp_5_excluded/                    # Campaign 5 held out
│   ├── camp_detector_5_epoch_1.pth
│   ├── camp_detector_5_epoch_2.pth
│   ├── camp_detector_5_epoch_3.pth
│   └── ep_3/
│       └── eval_1/
└── camp_8_excluded/                    # Campaign 8 held out (if subset=3)
    ├── camp_detector_8_epoch_1.pth
    ├── camp_detector_8_epoch_2.pth
    ├── camp_detector_8_epoch_3.pth
    └── ep_3/
        └── eval_1/
```

## Expected Performance Progression

### Training Loss (Per Campaign)

- **Epoch 1**: Initial loss ~0.5-0.9 (varies by campaign complexity)
- **Epoch 2**: Loss should decrease to ~0.3-0.6
- **Epoch 3**: Further decrease to ~0.2-0.5

### Cross-Campaign Performance

#### camp_1_excluded (Large Fake Software Campaign)

- **Epoch 1**: Detection rate ~88-94% at 1% FPR
- **Epoch 2**: Detection rate ~92-97% at 1% FPR
- **Epoch 3**: Detection rate ~96-99% at 1% FPR

#### camp_5_excluded (Notification Stealing Campaign)

- **Epoch 1**: Detection rate ~82-89% at 1% FPR
- **Epoch 2**: Detection rate ~87-94% at 1% FPR
- **Epoch 3**: Detection rate ~92-98% at 1% FPR

#### camp_8_excluded (Service Sign-up Scam Campaign)

- **Epoch 1**: Detection rate ~85-91% at 1% FPR
- **Epoch 2**: Detection rate ~90-95% at 1% FPR
- **Epoch 3**: Detection rate ~94-99% at 1% FPR

## Key Metrics to Verify

### Training Log Verification

Check `training_log.log` for:

- Successful campaign data loading and balancing
- Consistent training across different campaign combinations
- Cross-campaign validation performance
- Campaign-agnostic feature learning progress

### Performance Metrics

Each `eval_metrics.txt` should contain:

- **Accuracy**: >0.92 by epoch 3 for held-out campaign
- **Precision**: >0.92 by epoch 3
- **Recall**: >0.92 by epoch 3
- **F1 Score**: >0.92 by epoch 3
- **AUC Score**: >0.99 by epoch 3
- **Detection Rate @ 1% FPR**: >0.92 by epoch 3

### Campaign Generalization

Expected patterns:

- **Attack Type Independence**: Performance across different BMA categories
- **Visual Pattern Recognition**: Common visual attack elements identified
- **Text Pattern Understanding**: Persuasive language patterns recognized
- **Cross-Category Robustness**: Consistent performance across attack strategies

## Understanding Epoch Selection

### Why Different Epochs for Different Campaigns

The verification demonstrates early training, but in full training:

- **Simple Campaigns**: May converge quickly (epochs 4-6)
- **Complex Campaigns**: May require more training (epochs 8-10)
- **Campaign Size Impact**: Larger campaigns may converge faster
- **Attack Sophistication**: More sophisticated attacks may need longer training

### Campaign Training Dynamics

The verification should show:

- **Attack Pattern Learning**: Model identifies common BMA characteristics
- **Feature Generalization**: Visual and textual features become campaign-agnostic
- **Consistent Learning**: Similar training patterns across campaign types

## Campaign Categories Reference

### BMA Attack Categories

1. **Fake Software Download**: 29 campaigns (largest category)
2. **Notification Stealing**: 7 campaigns
3. **Service Sign-up Scam**: 20 campaigns
4. **Scareware**: 9 campaigns
5. **Fake Lottery/Sweepstakes**: 6 campaigns
6. **Technical Support Scam**: 3 campaigns

### Verification Subset Campaigns

- **Campaign 1**: Large fake software download campaign
- **Campaign 5**: Notification stealing campaign
- **Campaign 8**: Service sign-up scam campaign

### Full Training Campaign Models

- `m_camp_1_ep6.pth` → Campaign 1 held out (epoch 6)
- `m_camp_2_ep6.pth` → Campaign 2 held out (epoch 6)
- `m_camp_3_ep5.pth` → Campaign 3 held out (epoch 5)
- `m_camp_4_ep10.pth` → Campaign 4 held out (epoch 10)
- `m_camp_5_ep7.pth` → Campaign 5 held out (epoch 7)
- `m_camp_6_ep4.pth` → Campaign 6 held out (epoch 4)
- `m_camp_7_ep6.pth` → Campaign 7 held out (epoch 6)
- `m_camp_8_ep8.pth` → Campaign 8 held out (epoch 8)
- `m_camp_9_ep9.pth` → Campaign 9 held out (epoch 9)
- `m_camp_10_ep7.pth` → Campaign 10 held out (epoch 7)

## Troubleshooting

### Common Issues

1. **Data Imbalance**: Different campaigns may have varying sample sizes
2. **Feature Complexity**: Some attack patterns may be more complex to learn
3. **Convergence Variation**: Different campaigns may converge at different rates
4. **Generalization Challenges**: Ensure campaign-agnostic features are learned

### Expected Warnings

Normal warnings that may appear:

- Campaign data loading notifications
- Class imbalance handling messages
- Cross-campaign evaluation progress
- Feature extraction status updates

## Validation Criteria

The campaign training verification is successful if:

1. **Training Completes**: All campaign cycles finish without errors
2. **Cross-Campaign Performance**: >92% detection rate on held-out campaigns
3. **Consistent Generalization**: Similar performance patterns across attack types
4. **Files Generated**: All expected output files for each campaign split
5. **Attack Independence**: Good performance regardless of BMA category

## Full Campaign Training Expectations

If running full campaign training (all 10 campaigns):

- **Total Runtime**: 25-35 hours on modern GPU
- **Individual Convergence**: Each campaign typically converges by epoch 4-10
- **Global Performance**: >99% detection rate across all held-out campaigns
- **Attack Robustness**: Consistent performance across all BMA categories

## Key Insights

This verification demonstrates:

- **Attack Type Independence**: Models work across different BMA categories
- **Visual Pattern Generalization**: Recognition of common visual attack elements
- **Text Pattern Understanding**: Understanding of persuasive language patterns
- **Cross-Category Robustness**: Performance maintained across diverse attack strategies

## Expected Convergence Patterns

### Fast Converging Campaigns

- **Simple Attack Patterns**: May converge quickly (epochs 4-6)
- **Large Campaigns**: More data may lead to faster convergence
- **Common Patterns**: Shared BMA elements facilitate generalization

### Slower Converging Campaigns

- **Complex Attack Patterns**: May require more training (epochs 8-10)
- **Diverse Campaigns**: Complex patterns may need more epochs
- **Specialized Attacks**: Unique characteristics may affect optimal epochs

## Campaign Characteristics Impact

### Large Campaigns (e.g., Fake Software)

- **Data Abundance**: More training samples available
- **Pattern Consistency**: Common visual and textual patterns
- **Faster Convergence**: Typically converge by epochs 5-7

### Specialized Campaigns (e.g., Tech Support Scams)

- **Unique Patterns**: Distinctive attack characteristics
- **Complex Learning**: May require more epochs (8-10)
- **Sophisticated Attacks**: Advanced persuasion techniques

### Diverse Campaigns (e.g., Service Sign-ups)

- **Variable Content**: Wide range of attack presentations
- **Moderate Convergence**: Typically converge by epochs 6-8
- **Balanced Complexity**: Mix of simple and complex patterns

## Attack Pattern Analysis

The verification should reveal:

- **Visual Commonalities**: Shared design elements across campaigns
- **Textual Patterns**: Common persuasive language structures
- **Behavioral Cues**: Consistent manipulation techniques
- **Cross-Campaign Features**: Generalizable attack characteristics

The campaign training verification validates that the campaign-agnostic approach successfully enables generalization to never-before-seen BMA campaigns, supporting the RQ3 claims about cross-campaign detection capabilities.
