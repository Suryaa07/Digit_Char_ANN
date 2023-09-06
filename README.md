

# Character and Digit Recognition with Artificial Neural Network (ANN)

## Overview

This repository showcases a simple ANN-based model for character and digit recognition. The model is trained on the MNIST dataset for recognizing digits and extended to recognize custom characters. This README provides an overview of the project, its components, and how to use it.

## Prerequisites

Make sure you have the following libraries and frameworks installed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-Learn (for label encoding and data splitting)

## How it Works

1. **Data Loading**: The MNIST dataset, containing images of handwritten digits, is loaded. Additionally, a custom dataset of characters is generated with random data.

2. **Data Preprocessing**: Both datasets are preprocessed by normalizing pixel values to the range [0, 1].

3. **Model Architecture**: A simple ANN is defined using Keras. It consists of a flattening layer, a dense hidden layer with ReLU activation, and an output layer with softmax activation for multi-class classification.

4. **Model Compilation**: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

5. **Training**: The model is trained on a combination of the MNIST digits dataset and custom character dataset. The training history is recorded for visualization.

6. **Evaluation**: The model's performance is evaluated on both the MNIST test dataset and the custom character dataset.

7. **Visualization**: The training and validation loss and accuracy are visualized using Matplotlib.

## Conclusion

This project demonstrates the implementation of an ANN for character and digit recognition using the MNIST dataset as well as custom characters. You can extend this project by fine-tuning the model, increasing the dataset size, or exploring more complex neural network architectures to improve recognition accuracy. This ANN serves as a basic example and starting point for more advanced image recognition tasks.
