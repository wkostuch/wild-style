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
'''
End of imports
'''


# Loads a model from a filepath and returns that loaded model
def import_model(filepath: str):
    model_handle = filepath
    model_module = hub.load(model_handle)
    return model_module


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
    filepath = "../models/fast_model"
    module = import_model(filepath)
    #img = get_styled_image(module, )

    print(type(module))


#Wow, we're being run on our own!
if __name__ == "__main__":
    main()