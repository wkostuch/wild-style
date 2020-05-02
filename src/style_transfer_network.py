# Author: Michael Jurkoic

# Based on this paper: https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import tensorflow as tf


# Custom Keras layer for the Res_block layers (there are five of them in the
# Zhou paper's network).
class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride):
        super(ResBlock, self).__init__()
        # Define a convolution layer and an activation layer to be used in call.
        self.res_block_conv = tf.keras.layers.Conv2D(filters=filters,\
            kernel_size=kernel_size, stride=stride)
        self.res_block_activation = tf.keras.layers.Activation('relu')
        
    def call(self, input):
        x = self.res_block_conv(input)
        x = self.res_block_activation(x)
        x = self.res_block_conv(x)
        # Implement the shortcut.
        x = tf.keras.layers.Add([x, input])
        x = self.res_block_activation(x)
        return x


# Builds the style transfer network, but doesn't train it.
def build_model(output_activation='sigmoid'):
    """
    Build the style transfer network initialized with random weights.

    Parameters
    - output_activation (string): activation function to be used on the output
    layer.  Defaults to sigmoid.

    Returns
    - style_model: TensorFlow Keras sequential CNN model.

    Parameters for convolution layers are taken from Zhou et. al. (2019).
    """

    # Initialize a sequential model.
    style_model = tf.keras.Sequential()

    # Add the first three convolution layers and activations.
    # Conv1
    style_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 9),\
        stride=1))
    style_model.add(tf.keras.layers.Activation('relu'))
    # Conv2
    style_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),\
        stride=2))
    style_model.add(tf.keras.layers.Activation('relu'))
    # Conv2
    style_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),\
        stride=2))
    style_model.add(tf.keras.layers.Activation('relu'))
    # Res_block1
    style_model.add(ResBlock(filters=128, kernel_size=(3, 3), stride=1))
    # Res_block2
    style_model.add(ResBlock(filters=128, kernel_size=(3, 3), stride=1))
    # Res_block3
    style_model.add(ResBlock(filters=128, kernel_size=(3, 3), stride=1))
    # Res_block4
    style_model.add(ResBlock(filters=128, kernel_size=(3, 3), stride=1))
    # Res_block5
    style_model.add(ResBlock(filters=128, kernel_size=(3, 3), stride=1))
    # Deconv1
    style_model.add(tf.keras.layers.Conv2DTranspose(filters=64,\
        kernel_size=(3, 3), stride=2))
    style_model.add(tf.keras.layers.Activation('relu'))
    # Deconv2
    style_model.add(tf.keras.layers.Conv2DTranspose(filters=32,\
        kernel_size=(3, 3), stride=2))
    style_model.add(tf.keras.layers.Activation('relu'))
    # Conv4
    style_model.add(tf.keras.layers.Conv2D(filters=3, kernel_stride=(9, 9),\
        stride=1))
    # Output layer activation.
    # Changing the activation function will change the output image quality.
    style_model.add(tf.keras.layers.Activation('sigmoid'))

    return style_model
