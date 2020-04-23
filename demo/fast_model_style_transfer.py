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


#Imports a model from a filepath and returns that loaded model
def import_model(filepath: str):
    hub_handle = filepath
    hub_module = hub.load(hub_handle)
    return hub_module





# A main() method to run when the file is run on its own
def main():
    filepath = "../models/fast_model"
    module = import_model(filepath)
    print(type(module))


if __name__ == "__main__":
    main()