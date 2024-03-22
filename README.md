
# Deep Learning Image Classification Project

## Overview
This project demonstrates a simple image classification application using deep learning. It includes a convolutional neural network (CNN) model built with TensorFlow and Keras, a Flask API to serve predictions, and a basic HTML frontend for user interaction.

## Requirements
- Python 3.6 or later
- TensorFlow 2.x
- Flask
- PIL (Python Imaging Library)
- NumPy

## Project Structure
- `deep_learning_training.py`: Script for model training and evaluation.
- `flask_api_for_image_classification.py`: Python script for the Flask API.
- `index.html`: HTML file for the frontend interface.
- `image_classification_model.h5`: Saved model file (generated after training).

## Setup and Running the Application
1. **Model Training**
   - Run the Python script (`deep_learning_training`) to train the model. 
   - The model will be saved as `image_classification_model.h5`.

2. **API Setup**
   - Ensure the saved model is in the same directory as the Flask script.
   - Run `flask_api_for_image_classification.py` to start the Flask server.

3. **Frontend Interface**
   - Open `index.html` in a web browser, under the "templates" directory, make sure to open from the URL provided by Flask, not a local File Explorer (Windows)
   - Upload an image to get the classification result.

## Usage
- The Flask server must be running to serve predictions.
- Through the HTML interface, upload an image and click 'Upload and Classify' to get the classification result.

## Model Info

The application uses a simple CNN model trained on the CIFAR-10 dataset. The model architecture consists of convolutional layers followed by max-pooling layers, a flattening step, and dense layers at the end.

Training
The model was trained with the following configuration:

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Training was done for 100 epochs with early stopping based on validation accuracy to prevent overfitting.

Technologies Used
TensorFlow and Keras for building and training the CNN model.
Flask for serving the model and handling web requests.
HTML/CSS for the frontend interface.
