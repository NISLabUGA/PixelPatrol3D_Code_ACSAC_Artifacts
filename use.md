# Intended Use and Limitations

## Intended Use

This artifact is intended for:

1. **Academic Research**: Reproducing and validating the results presented in the paper "PP3D: An In-Browser Vision-Based Defense Against Web Behavior Manipulation Attacks"

2. **Educational Purposes**: Understanding multimodal machine learning approaches for web security, particularly behavior manipulation attack detection

3. **Baseline Comparison**: Serving as a baseline for future research in web-based social engineering detection

4. **Method Extension**: Providing a foundation for researchers to build upon and extend the PP3D framework

## Appropriate Use Cases

- Reproducing paper results for peer review or verification
- Comparing against PP3D in academic publications
- Educational demonstrations of multimodal ML for security
- Research into web-based social engineering detection
- Development of improved BMA detection systems

## Limitations

### Technical Limitations

1. **Dataset Dependency**: The model's performance is tied to the specific BMA campaigns and benign websites in the training data. Performance may vary on significantly different web content.

2. **Resolution Constraints**: While resolution-agnostic, the model was trained on specific resolution ranges. Extreme resolutions may affect performance.

3. **Language Limitations**: The text component is primarily trained on English content. Performance on non-English BMAs may be reduced.

4. **Temporal Drift**: As web design trends and attack techniques evolve, model performance may degrade over time without retraining.

5. **Computational Requirements**: Real-time deployment requires significant computational resources, particularly for the multimodal inference pipeline.

### Methodological Limitations

1. **False Positives**: The system may flag legitimate websites that share visual or textual similarities with BMAs (e.g., legitimate lottery sites, software download pages).

2. **Adversarial Vulnerability**: Despite adversarial training, the system may still be vulnerable to sophisticated, targeted adversarial attacks.

3. **Campaign Coverage**: The model was trained on specific BMA categories. Novel attack types not represented in training data may be missed.

4. **Context Dependency**: The system analyzes individual web pages without considering user browsing context or session information.

### Deployment Limitations

1. **Privacy Considerations**: While designed for local inference, deployment in production environments requires careful consideration of user privacy and data handling.

2. **Performance Overhead**: The multimodal inference pipeline introduces latency that may affect user experience in real-time applications.

3. **Maintenance Requirements**: Production deployment requires ongoing model updates and monitoring to maintain effectiveness against evolving threats.

## Ethical Considerations

### Responsible Use

- This tool should be used to protect users from malicious content, not to restrict access to legitimate websites
- Deployment should include mechanisms for users to report false positives and override blocking decisions
- Regular auditing should be conducted to ensure the system doesn't exhibit bias against specific types of legitimate content

### Potential Misuse

- The techniques demonstrated could potentially be adapted by attackers to create more sophisticated evasion methods
- The dataset and models should not be used to generate or improve behavior manipulation attacks
- Care should be taken to prevent the system from being used for censorship or inappropriate content filtering

## Data Handling

### Training Data

- The BMA dataset contains examples of malicious web content that could be harmful if accessed directly
- Researchers using this data should take appropriate precautions to avoid exposure to malicious content
- The dataset should not be redistributed without proper safeguards and ethical review

### User Data

- In deployment scenarios, the system processes user browsing data
- Appropriate privacy protections must be implemented
- Users should be informed about data processing and given control over the system's operation

## Reproducibility Notes

### Expected Variations

- Results may vary slightly due to:
  - Hardware differences (GPU vs CPU, different GPU models)
  - Software version differences in dependencies
  - Random seed variations in data loading and model initialization
  - Floating-point precision differences across platforms

### Validation Criteria

- Results within 2% of reported metrics should be considered successful reproduction
- Relative performance rankings between different evaluation scenarios should be preserved
- Overall trends and conclusions should remain consistent

## Support and Maintenance

### Artifact Maintenance

- This artifact represents a snapshot of the research at the time of publication
- No ongoing maintenance or updates are guaranteed
- Users should verify compatibility with current software versions

### Community Contributions

- Researchers are encouraged to share improvements and extensions
- Bug reports and fixes are welcome through appropriate academic channels
- Derivative works should properly cite the original research

## Disclaimer

This artifact is provided for research and educational purposes only. The authors make no warranties about the suitability of this software for any particular purpose and disclaim all liability for any damages arising from its use. Users are responsible for ensuring appropriate and ethical use of this artifact.
