# Author: Michael Jurkoic
# Last updated: 2020

# Code in this function is borrowed from the tutorials and code available at
# https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb


import functools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backed as K
import time


# Make sure eager execution is enabled for best performance
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


# Global variables
content_path = "images/"
style_path = "images/"

# The style transfer technique uses a pretrained image classification neural
# network from the Oxford Visual Geometry Group (VGG).  The technique pulls the
# image from certain layers in the model, getting it at different levels of
# convolution.  We define those levels here.

# Tweaking these parameters would be a good way to play with the results; try
# combining style and content images from different layers.

# Content layer from which we will pull our feature maps.
content_layers = ['block5_conv2']

# Style layers we're interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# Get the number of layers needed
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# Accepts a filepath as a string and returns an image.
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    length = max(img.size)
    scale = max_dim / length
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),\
        Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    # Broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


# Accepts an image and a title and displays the image using matplotlib
def imshow(img, title=None):
    # Get rid of the batch dimension
    out = np.squeeze(img, axis=0)
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


# Accepts an image and returns an image after some preprocessing.
# Uses the VGG training process built into TensorFlow.
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


# Accepts an image and returns another image after some deprocessing.
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of\
        dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
    # Reverse the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8)')


# The style transfer algorithm uses the VGG19 model.  We have to set it up in a
# way that lets us access the intermediate layers.
# This function returns a keras model that takes image inputs and outputs the
# intermediate layers we want to use for style and content.
def get_model():
    # Load the VGG19 model trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False,\
        weights='imagenet')
    vgg.trainable = False
    # Get the output layers corresponding to style and content.
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build the model
    return models.Model(vgg.input, model_outputs)


# Compute content loss by Euclidean distance.  See the linked tutorial at the
# top of this file for a thorough mathematical description.
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


# This function is a helper to the style loss function.
def gram_matrix(input_tensor):
    # Make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


# The style loss function is more complicated than the content loss function.
# It takes contribution from the several style layers.  In the basic
# implementation, each layer is weighted equally, but we could change the
# weights to change the output.  See the tutorial for the math.
# Expects two images as input.
def get_style_loss(base_style, gram_target):
    # Height, width, number of filters of each layer
    # Scale the loss at a given layer by the size of the feature map and the
    # number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


# Computes the feature representations for style and content.  It loads the
# images and feeds them through the VGG network to get the outputs of the
# intermediate layers.
# Accepts the model to be used, the content image, and the style image.
# Returns the style and content features.
def get_feature_representations(model, content_path, style_path):
    # Load the images
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # Compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations.
    style_features = [style_layer[0] for style_layer\
        in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer\
        in content_outputs[num_content_layers:]]
    return style_features, content_features


# Computes total loss.
# Accepts a model, the weights to assign each loss function contribution (tweak
# these to play with results), the base image, the gram matrices corresponding
# to the style layers we are interested in, and the outputs from the content
# layers we're interested in.
# Returns total loss, style loss, content loss, and total variational loss
def compute_loss(model, loss_weights, init_image, gram_style_features,\
    content_features):
    style_weight, content_weight = loss_weights
    # Feed the base image through our model.  This will obtain the content and
    # style representations from teh desired layers.
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0
    # Accumulate style losses from all layers
    # For now, weight each layer's contribution equally
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features,\
        style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0],\
            target_style)
    # Accumulate content losses form all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features,\
        content_output_features):
        content_score += weight_per_content_layer\
            * get_content_loss(comb_content[0], target_content)
    style_score *= style_weight
    content_score *= content_weight
    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


# Computer the loss gradients (for backpropogation)
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def main():
    # Get an image for content and an image for style
    content = load_img(content_path).astype('uint8')
    style = load_img(style_path).astype('uint8')


if __name__ == "__main__":
    main()
