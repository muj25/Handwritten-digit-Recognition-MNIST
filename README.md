# Handwritten-digit-Recognition-MNIST
This Handwritten Digit Recognition app allows users to draw digits on a canvas and predicts them using a CNN model. Built with `tkinter` and `customtkinter`, it features real-time predictions, bounding boxes, and a clear button, offering a modern, user-friendly interface with theme customization.

## Project Overview
### Data Preparation:

The MNIST dataset is loaded and preprocessed.
Images are reshaped and normalized for both CNN and Random Forest models.
The dataset is split into training, validation, and test sets.

### Convolutional Neural Network (CNN):
A CNN model is defined with convolutional, max-pooling, dropout, and dense layers.
The model is compiled with the Adam optimizer and categorical crossentropy loss function.
Data augmentation is applied using ImageDataGenerator for training.
The model is trained with callbacks for early stopping and model checkpointing.
The trained model is evaluated on the test set, and its performance is assessed with accuracy and R2 score.
A confusion matrix and accuracy plot are generated, and predictions on random test images are displayed.

### Random Forest Classifier:
The data is flattened and normalized using StandardScaler.
A Random Forest classifier is trained on the flattened training data.
The model is evaluated on the test set, and its performance is assessed with accuracy and R2 score.
A confusion matrix is generated to visualize the model's performance.
A single prediction is made on a random image from the test set, and the image along with the prediction is displayed.
The Random Forest model and scaler are saved using joblib.


### Handwritten Digit Recognition GUI
This GUI application allows users to draw handwritten digits and recognize them using a pre-trained Convolutional Neural Network (CNN). Built with tkinter and customtkinter, the application provides an interactive interface for digit recognition.

Features
Drawing Canvas: A large drawing area where users can draw digits.
Prediction: Button to trigger digit recognition on the drawn content.
Clear Canvas: Button to clear the drawing area and reset the application.
Result Display: Label to show the recognized digits and any bounding boxes drawn around detected digits.
