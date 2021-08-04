# Fashion-MNIST
Creating a web interface for image recognition using the trained model Fashion Mnist

# Build the Model

fashionMNIST.ipynb is the notebook where the Fashion MNIST dataset is uploded and the Convolutional Neural Network model was build & train using Keras API.

For tracking - Weights & Biases was used in the nootebook.

# Fask API

The trained model wrapped with Flask API.

model.py load the model, preprocess the images in order to be used by that model, and make predictions.

upload.py is responsible for running the API. It interacts with the web page where the user can upload an image.

# Containerize Docker

requirements.txt contains packages that were installed for the project.



