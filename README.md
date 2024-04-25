# DeepPermNet: Jigsaw Puzzle Solver
This repository contains the implementation of a DeepPermNet model for solving jigsaw puzzle problems on the CIFAR-10 dataset, as described in the paper "DeepPermNet: Visual Permutation Learning" by Rodrigo Santa Cruz et al.

## Abstract
We constructed a DeepPermNet to solve jigsaw puzzle problems, achieving a 95.67% accuracy rate on the test set after extensive exploration and optimization.

## Key Components
- **AlexNet**: A convolutional neural network used as a submodel to process and extract features from segmented sub-images.
- **DeepPermNet**: The main network architecture that predicts the permutation matrix to reconstruct the original order of shuffled sub-images.
- **Cosine Annealing**: An optimization algorithm used to adjust the learning rate for model stability and convergence.
- **Cross Entropy**: A measure used to calculate the loss value for the classification problem.

## Network Architecture
1. **AlexNet Submodel**: Processes input images and outputs a feature tensor.
2. **Batch Normalization Layer**: Normalizes features for improved training stability.
3. **MLP Classifier**: A multi-layer perceptron that maps features to output classes.
4. **L2 Regularization Layer**: Applies L2 regularization to constrain output values.

## Optimization Measures
- **Batch Normalization**: Improved accuracy and reduced training time.
- **Epoch Selection**: Determined optimal number of epochs for efficiency and accuracy.
- **Learning Rate Adjustment**: Implemented Cosine Annealing for dynamic learning rate adjustment.
- **Gradient Descent Algorithms**: Compared SGD and Adam, with SGD outperforming in both accuracy and training time.

## Results
- **Accuracy**: The model achieved a maximum accuracy of 95.67% in 100 epochs.
- **Convergence**: The loss steadily decreased and the model converged effectively.

## Conclusion
Significant optimizations were made to the base model, leading to high accuracy in solving jigsaw puzzles. However, there is potential for further improvement through deeper understanding and research.

## References
- [DeepPermNet: Visual Permutation Learning](https://arxiv.org/abs/1704.02729)

## Authors
- Bolun Zhang
- Shanghai Jiao Tong University

## Contact
For any questions or collaborations, please reach out to zbl1677590857@foxmail.com
