# A file to consolidate useful methods.

import cv2
import unittest
import tensorflow as tf
from PIL import Image
import numpy as np


# Imports an image from a filepath given as a string, and returns it as an
# OpenCV image.
def import_image(path):
    """Read an image into a cv2 object.

    Parameters:
    - path (string): Filepath to an image

    Returns:
    - cv2 image

    Supported file types:
    - .png
    """

    # List of supported file extensions as a tuple.
    # Must be a tuple because str.endswith accepts a tuple to check for one of
    # multiple possible extensions.
    extensions = ('.png')

    # Throw an exception if the file referenced isn't one of the supported file
    # types.
    try:
        assert path.endswith(extensions)
    except AssertionError:
        print("Filetype must be one of: " + extensions)
    
    # Import the image as a color image.
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    return img


#Takes a tensor and returns one that can be displayed via Pillow
def tensor_to_image(tensor):
    """
    Transforms a tensor into one that can be dispalyed via Pillow

    Parameters
    - tensor

    Returns
    - tensor
    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


#Takes an image-tensor, makes it displayable, and displays it via Pillow
def imshow(image, title=None):
    """
    Takes a tensor and displays it as an image.

    Note: this method calls tensor_to_image, so you don't have to
          preprocess by calling that on the tensor before giving
          it to this method
    """
    if len(image.shape) >= 3:
        image = tf.squeeze(image, axis=0)
    img = tensor_to_image(image)
    if title:
        img.show(title=title)
    else: 
        img.show(title="Picture")
