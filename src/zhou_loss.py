# Author: Michael Jurkoic

# Based on this paper by Zhou, et. al.: 
# https://www.mdpi.com/2071-1050/11/20/5673/htm

# Special thanks to Dr. Rob Hochberg for help understanding the contents of the
# paper.

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import numpy as np
import tensorflow as tf


# The only two functions intended to be used by other modules are content_loss
# and total_loss.  All other functions are not intended for use outside this
# module.


# Define the content loss function (equation 5 in the Zhou paper).
def content_loss(content_img, stylized_img):
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


# Function for computing the total variance loss to imcrease the smoothness of
# the stylized image.  Based on Zhou equation 9.
def _total_variance_loss(stylized_img):
    """
    Computes total variance loss of an image.

    Parameters
    - stylized_img (tensor): Image produced by the stylization network.

    Returns
    - tv_loss (float): Loss calculated by this function.

    The total variance loss is used so smooth the stylized image generated by
    the style transfer network.  This function implements equation 9 in the Zhou
    paper.
    """

    # We have to do an internal sum over each tensor, subtracting each pixel
    # from the one before it along rows and columns.  So we'll do it cleverly by
    # copying the input tensor, offsetting the copies, and doing some element by
    # element subtraction before taking the norm squared.

    # Get the computation along the height axis.
    img1 = tf.identity(stylized_img)[1:, :, :]
    img2 = tf.identity(stylized_img)[:-1,:, :]
    height_diff = (img2 - img1)
    height_sum = tf.norm(height_diff)**2

    # Get the computation along the width axis.
    img3 = tf.identity(stylized_img)[:, 1:, :]
    img4 = tf.identity(stylized_img)[:, :-1, :]
    width_diff = (img4 - img3)
    width_sum = tf.norm(width_diff)**2

    # Now get the total variance loss from the sum of the computations along the
    # height and width.
    tv_loss = height_sum + width_sum
    return tv_loss


# Get the total weighted loss (equation 10 in the Zhou paper).
# Image activations are performed in a different function before being fed into
# the loss computation.
def total_loss(content_image, content_stylized_image, style_images,\
    style_stylized_images, simple_stylized_image, total_loss_weights,\
    style_loss_weights=[0.0]):
    """
    Computes the total loss as a weighted sum of content, style, and total
    variance loss.

    Parameters
    - content_image (tensor): The activation of the content image.
    - content_stylized_image (tensor): The stylized image activated by the same
    VGG19 layer as the activated content image.
    - style_images (list of tensors): A list of activations of the style image.
    - style_stylized_images (list of tensors): A list of activations of the
    generated stylized image activated by the same VGG19 layers as the 
    activated style images.  This list mus
    - simple_stylized_image (tensor): The unactivated stylized image; used to
    compute the total variance loss.
    - total_loss_weights (tuple of floats): A 3-tuple of weights of the form (alpha, 
    beta, gamma) where alpha is the weight of the content loss, beta is the 
    weight of the style loss, and gamma is the weight of the total variance 
    loss.
    - style_loss_weights (optional list of floats): An optional list of weights
    to be used in weighting the style loss computation.  The list must be the
    same size as the style_images list, or an exception will be thrown.  By
    default, each style layer will be weighted equally in the style loss
    function.

    Returns
    - total_loss (float): The computed total loss.

    This function implements equation 10 from the Zhou paper, and uses all of
    the other loss functions.
    """

    # Split out the weights.
    alpha, beta, gamma = total_loss_weights

    # Get the weighted content loss.
    content_loss = _content_loss(content_image, content_stylized_images)
    weighted_content_loss = alpha * content_loss

    # Get the weighted style loss.
    style_loss = _total_style_loss(style_images, style_stylized_images,\
        style_loss_weights)
    weighted_style_loss = beta * style_loss

    # Get the weighted total variance loss.
    tv_loss = _total_variance_loss(simple_stylized_image)
    weighted_tv_loss = gamma * tv_loss

    # Compute and return the sum of the weighted losses.
    total_loss = weighted_content_loss + weighted_style_loss + weighted_tv_loss
    return total_loss
