# Expected Results for Training Verification: RQ1 & RQ4 Model

## Overview

This training verification demonstrates that the training process for the RQ1 & RQ4 model works correctly and shows why epoch 4 was selected for the main evaluation claims.

## Expected Training Behavior

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

## Expected Performance Progression

### Training Loss

- **Epoch 1**: Initial loss ~0.5-0.8 (depends on initialization)
- **Epoch 2**: Loss should decrease to ~0.3-0.5
- **Epoch 3**: Further decrease to ~0.2-0.4

### Validation Performance (RQ1 Test Set)

- **Epoch 1**: Detection rate ~85-95% at 1% FPR
- **Epoch 2**: Detection rate ~90-98% at 1% FPR
- **Epoch 3**: Detection rate ~95-99% at 1% FPR

### Validation Performance (RQ4 Test Set)

- **Epoch 1**: Detection rate ~80-90% at 1% FPR
- **Epoch 2**: Detection rate ~85-95% at 1% FPR
- **Epoch 3**: Detection rate ~90-97% at 1% FPR

## Key Metrics to Verify

### Training Log Verification

Check `training_log.log` for:

- Decreasing training loss over epochs
- Stable learning rate and gradient updates
- Successful batch processing without errors
- Validation evaluation completion

### Performance Metrics

Each `eval_metrics.txt` should contain:

- **Accuracy**: >0.95 by epoch 3
- **Precision**: >0.95 by epoch 3
- **Recall**: >0.95 by epoch 3
- **F1 Score**: >0.95 by epoch 3
- **AUC Score**: >0.99 by epoch 3
- **Detection Rate @ 1% FPR**: >0.90 by epoch 3

### ROC Curves

Each `roc_curve.png` should show:

- Strong performance with AUC >0.99
- Steep initial rise indicating good discrimination
- Performance improvement across epochs

## Understanding Epoch Selection

### Why Epoch 4 in Full Training

The verification demonstrates the training process, but in full training:

- **Epoch 4** typically shows peak performance
- **Validation performance** plateaus or slightly decreases after epoch 4
- **Generalization** is optimal at epoch 4 across both RQ1 and RQ4
- **Overfitting** may begin after epoch 4

### Training Dynamics

The verification should show:

- **Rapid Initial Learning**: Major improvements in epochs 1-2
- **Fine-tuning**: Gradual improvements in epoch 3
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

1. **Training Completes**: All 3 epochs finish without errors
2. **Loss Decreases**: Training loss shows downward trend
3. **Performance Improves**: Validation metrics improve over epochs
4. **Files Generated**: All expected output files are created
5. **Metrics Reasonable**: Performance values are within expected ranges

## Full Training Expectations

If running full training (10+ epochs):

- **Peak Performance**: Typically achieved at epoch 4
- **Final Model**: `artifacts/models/rq1_rq4/m_ep4.pth`
- **RQ1 Performance**: >99% detection rate at 1% FPR
- **RQ4 Performance**: >97% detection rate at 1% FPR

This verification demonstrates that the training process works correctly and provides insight into why epoch 4 was selected for the main evaluation claims.
