# A file to consolidate useful methods.
# TODO: Add tensor to OpenCV image conversion.

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


# Loads an image into a tensor
def load_img_as_tensor(path_to_img, dimension=512):
    """Loads the image from the given filepath as a tensor object.

    Parameters:
    - path_to_img (string): Filepath to an image
    - dimension (int): Maximum size of the shortest side of the image (default:
    512)

    Returns:
    - tensor

    Supported file types for loading images:
    - .png
    """
    max_dim = dimension
    # Read the image into a tensor
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize the image to fit in a square of sides given by the dimension 
    # paramater, preserving its original aspect ratio.
    img = tf.image.resize(img, (dimension, dimension),\
        preserve_aspect_ratio=True)

    return img


# Takes a tensor and returns a square, centered slice of that tensor.
def squarify_tensor(tensor):
    """
    Returns the a square of the given tensor.

    Parameters
    - tensor: A 3D image tensor.
    """

    # Get the length of the shortest of the width/height dimensions.
    shape = tf.cast(tf.shape(tensor)[:2], tf.float32)
    short_side = min(shape)
    radius = short_side // 2
    # Find the center of the tensor.
    center = [shape[0] // 2, shape[1] // 2]
    # Get the start and end indices of the slice.
    vert_start = center[0] - radius
    vert_end = vert_start + short_side
    horiz_start = center[1] - radius
    horiz_end = horiz_start + short_side
    # Make the image a 4D tensor if needed.
    if len(tf.shape(tensor)) == 3:
        tensor = tf.expand_dims(tensor, 0)
    # Slice out the square.
    square_tensor = tf.image.crop_and_resize(tensor,\
        [[vert_start, vert_end, horiz_start, horiz_end]], [0],\
            [short_side, short_side])[:1, :, :, :]
    return square_tensor


# Same thing as squarify_tensor, but for OpenCV and Numpy objects rather than
# TensorFlow objects.
# This function works for sure, squarify_tensor hasn't been fully tested yet.
def squarify_image(img, dim):
    """
    Crops out a centered square from an OpenCV image.

    Parameters
    - img (OpenCV image): Image to be cropped.
    - dim (int): Length of the sides of the output image.

    Returns
    - sqr_img (OpenCV image): Cropped square from the center of the input image
    with dimensions (dim, dim, 3).

    This function takes the largest possible square that can fit in the original
    image, then resizes it to have sides of length given by dim.
    """

    # Get the length of the shorter side, and use that as the length reference
    # for getting a square from the image.
    shape = img.shape[:2]
    ref_side = min(shape)
    # Make the length an even number.
    if ref_side % 2 == 1:
        ref_side = ref_side - 1
    # Get the start and end indices across both dimensions.
    height_start = (shape[0] - ref_side) // 2
    height_end = height_start + ref_side
    width_start = (shape[1] - ref_side) // 2
    width_end = width_start + ref_side
    # Slice a square out of the center of the input image.
    sqr_img = img[height_start:height_end, width_start:width_end, :]
    # Resize the image to the desired output size.
    # Let the interpretation method depend on whether the image will be expanded
    # or contracted.
    # Use INTER_AREA interpolation for shrinking.
    if ref_side >= dim:
        sqr_img = cv2.resize(sqr_img, (dim, dim), interpolation=cv2.INTER_AREA)
    # Use INTER_CUBIC interpolation for expanding.
    elif ref_side < dim:
        sqr_img = cv2.resize(sqr_img, (dim, dim), interpolation=cv2.INTER_CUBIC)
    # Return the resized image.
    return sqr_img



# Takes a tensor and returns one that can be displayed via Pillow
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


# Converts an OpenCV image into a tensor.
def cv2_image_to_tensor(img):
    """
    Convert an OpenCV image into a TensorFlow tensor.

    Parameters
    - image (cv2 image): 3-color BGR image.

    Returns
    - tensor: 3-color RGB tensor of floats.
    """
    
    # Convert the BGR image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the RGB OpenCV image to a tensor filled with floats.
    tensor = tf.convert_to_tensor(rgb_img, dtype='float32')
    return tensor


#Takes an image-tensor, makes it displayable, and displays it via Pillow
def display_tensor_as_image(image, title=None):
    """
    Takes a tensor and displays it as an image using the system's
    default image-viewer.

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


# Returns the size of 256, 512, or 1024, that the input value is closest to.
def closest_valid_size(input_dim):
    """
    Get the closest valid Zhou-compatible size to the input dimension.

    Parameters
    - input_dim (int)

    Returns
    - closest_size (int)
    """

    sizes = {"1":256, "2":512, "3":1024}
    size_dists = []
    for i in range(1, 4):
        size_dists.append(abs(sizes[str(i)] - input_dim))
    closest_size = min(size_dists)

    return closest_size


def get_square_center(img, size=0):
    """
    Crop the input image to a square of a size compatible with the Zhou style
    transfer network, while preserving as much image content as possible.

    Parameters
    - img (cv2 image): Image of any size and aspect ratio.
    - size (int): Size of the output image. 1 for 256x256, 2 for 512x512, and 3
    for 1024x1024.  Any other number will set output image to default size.

    Returns
    - square_img (cv2 image): Square image of the size specified by 'size' 
    parameter.  If no size is specified, the function determines an output size
    by finding the size nearest to the dimensions of the cropped square portion
    of the original image.

    This function first finds the largest area square that fits within the
    original image, slices that square out of the input image, then resizes it
    to a size compatible with the Zhou style transfer network.
    """

    # Find the coordinates of the center of the input image.
    # Coordinates are given as (width, height).
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # The side length of the largest internal square is the smaller of the input
    # image's width/height dimensions.
    square_dim = min([img.shape[0], img.shape[1]])
    edge_dist = square_dim // 2

    # Slice a square of the proper dimension out of the input, preserving the
    # number of input channels.
    square_img = img[center[1]-edge_dist:center[1]+edge_dist,\
        center[0]-edge_dist:center[0]+edge_dist, :]

    # Determine the size of the output image.
    output_dim = 0
    if size == 1:
        output_dim = 256
    elif size == 2:
        output_dim = 512
    elif size == 3:
        output_dim = 1024
    else:
        output_dim = closest_valid_size(square_dim)

    # Resize the output image to the appropriate dimensions.
    square_img = cv2.resize(square_img, (output_dim, output_dim))

    return square_img


def tensor_to_cv2(tensor):
    """
    Convert an RGB image tensor to a BGR OpenCV image.

    Parameters
    - tensor (TensorFlow tensor)

    Returns
    - image (OpenCV image)
    """
    image = tensor.numpy()
    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv_img
