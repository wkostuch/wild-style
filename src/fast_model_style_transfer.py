# The model used in this code is from:
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb


'''
Set up all the necessary imports 
'''
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from PIL import Image

import time

import functools

import cv2

import method_library as meth
'''
End of imports
'''


#Get the file-paths for our images
content_path = "../images/content/rome_waterfront.png"
style_path =  "../images/style/starry_night.png"

#Load the image into a tensor
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #Compute new dimensions for the image
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    #Resize the image-tensor
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# Loads a model from a filepath and returns that loaded model
def import_model(filepath: str):
    model_handle = filepath
    model = hub.load(model_handle)
    return model


# Returns the outputs of the model with two inputs: content_image and style_image
def get_model_results(model, content_image, style_image):
    outputs = model(content_image, style_image)
    return outputs


# Returns the styled image from the outputs of the model
def get_styled_image_from_outputs(outputs):
    stylized_image = outputs[0]
    return stylized_image


# Returns the stylized image given a model, content_image, and a style_image
def get_styled_image(model, content_image, style_image):
    outputs = get_model_results(model, content_image, style_image)
    styled_image = get_styled_image_from_outputs(outputs)
    return styled_image

# Returns the stylized image given a content_image and a style_image
# using the default model
def get_styled_image(content_image, style_image):
    filepath = "..models/fast_model"
    model = import_model(filepath)
    styled_image = get_styled_image(model, content_image, style_image)
    return styled_image


'''
Need to make methods for getting the image from filepath, turning it into a tensor, etc
'''


# A main() method to run when the file is run on its own
def main():
    filepath = "../models/fast_model/"
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    model = import_model(filepath)
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    print(type(stylized_image))
    meth.imshow(stylized_image)
    #img = get_styled_image(module, )

    print(type(model))
    #print(model.summary())


#Wow, we're being run on our own!
if __name__ == "__main__":
    main()