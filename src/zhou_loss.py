# Author: Michael Jurkoic

# Based on this paper by Zhou, et. al.: 
# https://www.mdpi.com/2071-1050/11/20/5673/htm

# Special thanks to Dr. Rob Hochberg for help understanding the contents of the
# paper.

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import numpy as np
import tensorflow as tf


# Define the content loss function (equation 5 in the Zhou paper).
def _content_loss(content_img, stylized_img):
    """
    Compute the content loss between the content image and the stylized image.

    Parameters
    - content_img (tensor): The activated content image.
    - stylized_img (tensor): The activated image from the stylization network.

    Returns
    - content_loss (float): The output of the loss function.

    This function is an implementation of equation 5 in the Zhou paper.  Special
    thanks to Dr. Rob Hochberg for providing some of the Python implementation.
    """

    # Get the Euclidean distance between the two tensors.
    diff = content_img - stylized_img
    diff_norm = tf.norm(diff)**2
    # Divide that distance by the product of the content image's dimensions.
    content_loss = diff_norm.numpy() / tf.reduce_prod(content_img.shape).numpy()
    
    return content_loss


# Define the function that computes the Gram matrix.  According to the Zhou
# paper, the Gram matrix reflects the general style of the image, and is useful
# for comparing style features.
def _gram_matrix(image):
    """
    Compute the Gram matrix of the given image.

    Parameters
    - image (tensor): An image activated by the VGG19 network.

    Returns
    - gram (tensor): Gram matrix of the input image.

    This function implements equation 6 in the Zhou paper.  The Gram matrix is a
    good representation of an image's style, and is here used as a feature
    extractor.
    """

    # Use einsum to compute the Gram matrix.
    gram = tf.einsum('ijc,ijd->cd', image, image)
    return gram


# Define the style loss function (Zhou equation 7).
def _layer_style_loss(style_img, stylized_img):
    """
    Compute the style loss between two images.

    Parameters
    - style_img (tensor): The style image activated by VGG19.
    - stylized_img (tensor): The image produced by the stylization network.

    Returns
    - style_loss (float): The output of the loss function.

    This function implements equation 7 in the Zhou paper.
    """

    # Get the Euclidean distance between the Gram matrices of the two images.
    style_img_gram = _gram_matrix(style_img)
    stylized_img_gram = _gram_matrix(stylized_img)
    diff = stylized_img_gram - style_img_gram
    diff_norm = tf.norm(diff)**2
    # Divide by the product of the dimensions of the input images.
    style_loss = diff_norm.numpy() / tf.reduce_prod(style_img.shape).numpy()

    return style_loss


# Implement Zhou equation 8 to get the total style loss across all activation
# layers (i.e., get the style loss for all the images extracted from VGG19 for
# style loss computation).
def _total_style_loss(style_images, stylized_images, weights=[0.0]):
    """
    Computes the style loss across all activation layers.

    Parameters
    - style_images (list of tensors): List style image activations.
    - stylized_images (list of tensors): List stylized image activations.
    - weights (optional list of floats): List of weights for the corresponding
    activations in the input lists of tensors.  Defaults to a weight of 1.0 for
    each activation.

    Returns
    - total_style_loss (float): Output of the loss function

    This function implements equation 8 in the Zhou paper.  Image activations
    are tensors pulled from selected convolution layers in the VGG19 network.
    """

    # Make sure the input lists are the same length.
    try:
        assert len(style_images) == len(stylized_images)
    except AssertionError:
        print("The number of style image activations must equal the number "\
            + "of stylized image activations.")

    # If no weights are specified, set them all to 1.0.
    if weights == [0.0]:
        weights = [1.0 for _ in range(len(style_images))]

    # Make sure the number of weights matches the number of image activations.
    try:
        assert len(style_images) == len(weights)
    except AssertionError:
        print("The number of weights must equal the number of activations.")

    # Sum the weighted style losses of all the layers.
    total_style_loss = 0
    for i in range(len(style_images)):
        layer_loss = _layer_style_loss(style_images[i], stylized_images[i])
        weighted_layer_loss = layer_loss * weights[i]
        total_style_loss = total_style_loss + weighted_layer_loss

    return total_style_loss
