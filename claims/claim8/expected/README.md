# Expected Results for Training Verification: RQ3 Campaign Models

## Overview

This campaign training verification demonstrates that the leave-one-out campaign training process for RQ3 works correctly and shows how different epoch models were selected for different campaigns.

## Expected Training Behavior

### Training Time

- **Per Epoch**: ~5 minutes (V100 32GB GPU)
- **Verification (1-2 epochs, subset of campaigns)**: ~15-30 minutes (recommended for reviewers)
- **Full Training (10 campaigns, 10 epochs max)**: ~8.5 hours total (not necessary for verification)

### Leave-One-Out Campaign Training Progress

The verification run should demonstrate:

1. **Leave-One-Out Cycles**: Each campaign held out while training on others
2. **Campaign-Agnostic Processing**: Features generalize across attack patterns
3. **Cross-Campaign Evaluation**: Performance tested on held-out campaigns
4. **Model Checkpointing**: Separate models saved for each campaign split

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
└── camp_8_excluded/                    # Campaign 8 held out
    ├── camp_detector_8_epoch_1.pth
    ├── camp_detector_8_epoch_2.pth
    ├── camp_detector_8_epoch_3.pth
    └── ep_3/
        └── eval_1/
```

## Key Metrics to Verify

### Training Log Verification

Check `training_log.log` for:

- Successful campaign data loading and balancing
- Consistent training across different campaign combinations
- Cross-campaign validation performance
- Campaign-agnostic feature learning progress

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

- `m_camp_1_ep10.pth` → Campaign 1 held out (epoch 10)
- `m_camp_2_ep3.pth` → Campaign 2 held out (epoch 3)
- `m_camp_3_ep3.pth` → Campaign 3 held out (epoch 3)
- `m_camp_4_ep4.pth` → Campaign 4 held out (epoch 4)
- `m_camp_5_ep5.pth` → Campaign 5 held out (epoch 5)
- `m_camp_6_ep10.pth` → Campaign 6 held out (epoch 10)
- `m_camp_7_ep4.pth` → Campaign 7 held out (epoch 4)
- `m_camp_8_ep2.pth` → Campaign 8 held out (epoch 2)
- `m_camp_9_ep4.pth` → Campaign 9 held out (epoch 4)
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
2. **Consistent Generalization**: Similar performance patterns across attack types
3. **Files Generated**: All expected output files for each campaign split
4. **Attack Independence**: Good performance regardless of BMA category

## Verification vs. Full Training

### Recommended Verification (1-2 epochs, subset of campaigns)

- **Purpose**: Verify leave-one-out campaign training pipeline works correctly
- **Time**: ~15-30 minutes
- **Sufficient to demonstrate**: Campaign preprocessing, cross-campaign evaluation, model checkpointing
- **Cost-effective**: Minimal compute resources required

### Optional Full Training (10 campaigns, 10 epochs max)

- **Purpose**: Reproduce exact campaign model performance
- **Time**: ~8.5 hours total
- **Not necessary for verification**: Claims 1-4 already provide reproducibility testing
- **Individual Convergence**: Each campaign typically converges by epoch 4-10
- **Global Performance**: >95% detection rate across all held-out campaigns
- **Attack Robustness**: Consistent performance across all BMA categories

**Note**: Running 1-2 epochs on a subset of campaigns is sufficient to verify the campaign training process works correctly and is the recommended approach for reviewers. Full training is not necessary since claims 1-4 already handle reproducibility verification.

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
