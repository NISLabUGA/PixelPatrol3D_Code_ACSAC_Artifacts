# Expected Results for Training Verification: RQ1 & RQ4 Model

## Overview

This training verification demonstrates that the training process for the RQ1 & RQ4 model works correctly and shows why epoch 4 was selected for the main evaluation claims.

## Expected Training Behavior

### Training Time

- **Per Epoch**: ~15 minutes (V100 32GB GPU)
- **Verification (1-2 epochs)**: ~15-30 minutes (recommended for reviewers)
- **Full Training (4 epochs)**: ~1 hour (not necessary for verification)

### Training Progress

The verification run should demonstrate:

1. **Initialization**: Model and data loaders initialize successfully
2. **Training Loop**: Loss decreases over epochs with stable learning
3. **Validation**: Performance improves on RQ1 and RQ4 test sets
4. **Checkpointing**: Model files saved at each epoch
5. **Evaluation**: Comprehensive metrics computed and saved

### Expected Output Structure

```
artifacts/train_test/out/comb/
├── training_log.log                    # Detailed training progress
├── notes.txt                          # Configuration used
├── comb_detector_epoch_1.pth          # Epoch 1 checkpoint
├── comb_detector_epoch_2.pth          # Epoch 2 checkpoint
├── comb_detector_epoch_3.pth          # Epoch 3 checkpoint
├── ep_1/                              # Epoch 1 evaluation
│   ├── eval_1/                        # RQ1 test set results
│   │   ├── eval_metrics.txt
│   │   ├── roc_curve.png
│   │   ├── y_true.npy
│   │   ├── y_scores.npy
│   │   └── vis_conf/                  # Qualitative examples
│   └── eval_2/                        # RQ4 test set results
│       ├── eval_metrics.txt
│       ├── roc_curve.png
│       ├── y_true.npy
│       ├── y_scores.npy
│       └── vis_conf/
├── ep_2/                              # Epoch 2 evaluation
│   ├── eval_1/
│   └── eval_2/
└── ep_3/                              # Epoch 3 evaluation
    ├── eval_1/
    └── eval_2/
```

## Understanding Epoch Selection

The verification demonstrates the training process, but in full training:

- **Epoch 4** typically shows peak performance
- **Validation performance** plateaus or slightly decreases after epoch 4
- **Generalization** is optimal at epoch 4 across both RQ1 and RQ4
- **Overfitting** may begin after epoch 4

### Training Dynamics

The verification should show:

- **Rapid Initial Learning**: Major improvements in epochs 1-2
- **Fine-tuning**: Gradual improvements in epochs
- **Convergence Pattern**: Loss stabilization indicating approaching optimum

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size if GPU memory insufficient
2. **Slow Training**: Ensure GPU is being used (check CUDA availability)
3. **Poor Performance**: Verify dataset paths and data loading
4. **Missing Files**: Check that all required data files are present

### Expected Warnings

Normal warnings that may appear:

- Tokenizer warnings about sequence length
- CUDA initialization messages
- Model loading notifications

## Validation Criteria

The training verification is successful if:

1. **Training Completes**: All epochs finish without errors
2. **Loss Decreases**: Training loss shows downward trend
3. **Performance Improves**: Validation metrics improve over epochs
4. **Files Generated**: All expected output files are created
5. **Metrics Reasonable**: Performance values are within expected ranges

## Verification vs. Full Training

### Recommended Verification (1-2 epochs)

- **Purpose**: Verify training pipeline works correctly
- **Time**: ~15-30 minutes
- **Sufficient to demonstrate**: Loss decreases, checkpoints save, evaluation runs
- **Cost-effective**: Minimal compute resources required

### Optional Full Training (4 epochs)

- **Purpose**: Reproduce exact model performance
- **Time**: ~1 hour
- **Not necessary for verification**: Claims 1-4 already provide reproducibility testing
- **Peak Performance**: Typically achieved at epoch 4
- **Final Model**: `artifacts/models/rq1_rq4/m_ep4.pth`

**Note**: Running 1-2 epochs is sufficient to verify the training process works correctly and is the recommended approach for reviewers. Full training is not necessary since claims 1-4 already handle reproducibility verification.
