# Neural Network Implementations and Phoneme Recognition with MLPs

This repository contains solutions for two distinct tasks focused on neural network implementation and optimization. Each section provides details about the problem addressed, the methods implemented, and the outcomes.

---

## Part 1: Multilayer Perceptrons from Scratch

### Overview
In this task, we implemented a fully functional deep learning library, akin to PyTorch, named `MyTorch`. The library includes all core components required to build, train, and optimize Multilayer Perceptrons (MLPs). The implementation focuses on fundamental concepts like forward and backward propagation, loss calculation, optimization techniques, and regularization.

### Key Features
- **Core Components**:
  - Linear Layers
  - Activation Functions (Sigmoid, ReLU, Tanh, GELU, Softmax)
  - Loss Functions (Mean Squared Error, Cross-Entropy)
  - Batch Normalization
  - Stochastic Gradient Descent (SGD) with momentum
  
- **Models Built**:
  - MLP with 0 hidden layers
  - MLP with 1 hidden layer
  - MLP with 4 hidden layers

### Objective
The main goal was to provide an in-depth understanding of neural networks by implementing all components from scratch using only NumPy, without relying on frameworks like PyTorch or TensorFlow.

---

## Part 2: Speech Recognition with MLPs

### Overview
The focus of this task was to develop a Multilayer Perceptron (MLP) for phoneme state classification using speech data represented as mel spectrograms. The challenge involved building a robust model to achieve high accuracy on the phoneme classification task.

### Key Highlights
- **Dataset**: Mel spectrogram frames paired with phoneme state labels.
- **Model**: A feedforward neural network with hyperparameter tuning to improve performance.
- **Techniques**:
  - Context windowing to include surrounding frames for better predictions.
  - Time masking and Frequency masking for data augmentation, enhancing the model's generalization.
  - Experimentation with various architectures, activations, and optimizers.
  - Regularization techniques like dropout and weight decay.

### Result
The final model achieved an **accuracy of 86.3%**, demonstrating its effectiveness in phoneme state classification.

---
