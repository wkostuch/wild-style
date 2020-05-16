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
def _style_loss(style_img, stylized_img):
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
