# A file to consolidate useful methods.

import cv2
import unittest


# Imports an image from a filepath given as a string, and returns it as an
# OpenCV image.
def import_image(path):
    """Read an image into a cv2 object.

    Parameters:
    path (string): Filepath to an image

    Returns:
    cv2 image

    Supported file types:
    .png
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
