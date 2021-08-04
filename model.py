import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
import numpy as np


model = load_model('model_fashion2.h5', compile=False)

target_size = (28, 28)
classes = [
  'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def process_image(image):
    #image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image = image/255.0
    return image

    
def predict_class(image):
    y_prob = model.predict(image) 
    y_classes = y_prob.argmax(axis=-1)
    prediction = classes[np.argmax(y_prob)]
    percentage = '%.2f%%' % (y_prob[0,y_classes]*100)

    return prediction, percentage



if __name__ == '__main_':
    '''for test'''
    #load an image from file
    image = load_img('../image.jpg', target_size=(28, 28))
    image = prpcess_image(image)
    prediction, percentage = predict_class(image)
    print(prediction, percentage)
