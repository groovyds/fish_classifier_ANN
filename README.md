# Fish Species Classifier

Kaggle Fish Classifier Project Link:
https://www.kaggle.com/code/emircengel/fish-classifier

Data Link:
https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data

This repository contains a deep learning-based project for classifying different species of fish using a Artificial Neural Network (ANN). The dataset consists of fish images, and the goal is to identify the species from an input image.

## Table of Contents

- A
- b
- c


## Project Overview

The Fish Species Classifier is a machine learning project that uses a deep learning model to classify various fish species. The model is trained on a dataset of fish images, leveraging TensorFlow and Keras to build and train the network. The solution involves data preprocessing, model training, evaluation, and deployment steps.

## Dataset

The dataset contains 9 different fish types, and a "GT" version of the image which was pre-processed and they're organized in folders by species names.
For each class, there are 1000 augmented images and their pair-wise augmented ground truths.
Each class can be found in the "Fish_Dataset" file with their ground truth labels. All images for each class are ordered from "00000.png" to "01000.png".

For example, if you want to access the ground truth images of the shrimp in the dataset, the order should be followed is "Fish->Shrimp->Shrimp GT".

## Preprocessing Steps:

Resizing all images to a fixed dimension from (445x590x3) to (128x128x3).
Normalizing pixel values to the range [0, 1].
One-hot encoding the target labels.

## Model Architecture
The model is a simple Artificial Neural Network (ANN) designed for multi-class classification:

Input Layer: 
- Accepts a flattened image of shape (128, 128, 3).

Hidden Layers:
- Dense layers with ReLU activation functions.
- Dropout for regularization.

Output Layer: 
- Softmax activation for multi-class classification.

The architecture looks like this:

Input Layer: (128, 128, 3)
Flatten Layer
Dense Layer: 512 neurons, ReLU activation
Batch Normalization
Dropout: 0.3
Dense Layer: 256 neurons, ReLU activation
Batch Normalization
Dropout: 0.3
Dense Layer: 128 neurons, ReLU activation
Batch Normalization
Dropout: 0.3
Output Layer: 9, Softmax activation

## Results & Evaluation

Training Accuracy: Achieved approximately 96.49%.
Validation Accuracy: Achieved approximately 95.22%.
Test Accuracy: Evaluated at 95.88%.
Included visualizations such as loss and accuracy plots graphs, a classification report and a confusion matrix.

## Additional Evaluation

Visualised samples where the model was mistaken for further fine tuning potential.

## Future Work

Hyperparameter Tuning: Experiment with different model architectures, learning rates, activation functions and optimizers.
Data Augmentation: Apply additional data augmentation techniques to improve model generalization.
Model Improvements: Use more sophisticated models like CNNs for better accuracy.

