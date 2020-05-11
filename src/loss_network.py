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
def __compute_content_loss(content_img, stylized_img):
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
