# Expected Results for RQ5

## Overview

RQ5 tests the model's robustness against adversarial examples and demonstrates the effectiveness of adversarial training in improving robustness.

## Experimental Setup

- **Visual Perturbations**: PGD L-infinity attacks with ε ∈ {2, 4, 8, 16, 32}/255
- **Text Perturbations**: 5 levels of increasing severity
- **Test Levels**: 6 evaluation sets (clean + 5 adversarial levels)

## Expected Results Comparison

Based on Table 5 in the paper:

| Level | ε (pixel) | Before Adv Training | After Adv Training |
| ----- | --------- | ------------------- | ------------------ |
| clean | clean     | 1.0                 | 1.0                |
| 1     | 2/255     | 0.968               | 0.998              |
| 2     | 4/255     | 0.816               | 0.998              |
| 3     | 8/255     | 0.840               | 0.998              |
| 4     | 16/255    | 0.564               | 0.982              |
| 5     | 32/255    | 0.040               | 0.994              |

## Key Findings

1. **Before Adversarial Training**: Performance degrades significantly under attack

   - Clean performance: 100% DR@1%FPR
   - Severe degradation at higher attack strengths (4% at level 5)

2. **After Adversarial Training**: Robust performance maintained
   - Clean performance: 100% DR@1%FPR (preserved)
   - Strong robustness: >98% DR@1%FPR even at highest attack strength

## Output Structure

```
./out/
├── comb_no_adv/               # Before adversarial training
│   └── ep_1/
│       ├── eval_0/            # Clean examples
│       ├── eval_1/            # Level 1 adversarial
│       ├── eval_2/            # Level 2 adversarial
│       ├── eval_3/            # Level 3 adversarial
│       ├── eval_4/            # Level 4 adversarial
│       └── eval_5/            # Level 5 adversarial
└── comb_adv/                  # After adversarial training
    └── ep_1/
        ├── eval_0/            # Clean examples
        ├── eval_1/            # Level 1 adversarial
        ├── eval_2/            # Level 2 adversarial
        ├── eval_3/            # Level 3 adversarial
        ├── eval_4/            # Level 4 adversarial
        └── eval_5/            # Level 5 adversarial
```

## Key Metrics to Verify

1. **Before Adversarial Training** (comb_no_adv):

   - eval_0: DR@1%FPR = 1.0
   - eval_4: DR@1%FPR ≈ 0.564
   - eval_5: DR@1%FPR ≈ 0.040

2. **After Adversarial Training** (comb_adv):

   - eval_0: DR@1%FPR = 1.0
   - eval_4: DR@1%FPR ≈ 0.982
   - eval_5: DR@1%FPR ≈ 0.994

3. **Improvement Demonstration**:
   - Substantial robustness gains at all adversarial levels
   - No degradation in clean performance
   - Consistent high performance across attack strengths

## Adversarial Attack Details

- **Level 1**: Minor character-level noise, barely perceptible visual changes
- **Level 2**: Light paraphrasing + small visual perturbations
- **Level 3**: Moderate semantic shifts + noticeable visual artifacts
- **Level 4**: Strong semantic inversion + clearly visible perturbations
- **Level 5**: Extreme distortion + heavy visual noise
