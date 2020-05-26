# Author: Michael Jurkoic

# Based on this paper by Zhou, et. al.: 
# https://www.mdpi.com/2071-1050/11/20/5673/htm

# Special thanks to Dr. Rob Hochberg for help understanding the contents of the
# paper.

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import tensorflow as tf


# Custom Keras layer for the Res_block layers (there are five of them in the
# Zhou paper's network).
class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ResBlock, self).__init__()
        # Define a convolution layer and an activation layer to be used in call.
        self.res_block_conv = tf.keras.layers.Conv2D(filters=filters,\
            kernel_size=kernel_size, strides=strides)
        self.res_block_activation = tf.keras.layers.Activation('relu')
        
    def call(self, input):
        x = self.res_block_conv(input)
        x = self.res_block_activation(x)
        x = self.res_block_conv(x)
        # Implement the shortcut.
        x = tf.keras.layers.Add([x, input])
        x = self.res_block_activation(x)
        return x


class Stylizer(tf.keras.Model):
    """
    Custom Keras model for stylizing images.

    Constructor parameters
    - ouput_activation (string): Activation function to be used on the output
    layer.  Defaults to 'sigmoid'.

    Layer parameters are those given by the Zhou paper.
    """

    def __init__(self, output_activation='sigmoid', name='stylizer', **kwargs):
        super(Stylizer, self).__init__(name=name, **kwargs)
        # Define all the layers that will be needed.
        # Each convolution and deconvolution layer is distinct.
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 9),\
            strides=(1, 1), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),\
            strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),\
            strides=(2, 2), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=3, kernel_size=(9, 9),\
            strides=(1, 1), activation='relu')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=64,\
            kernel_size=(3, 3), strides=(2, 2), activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=32,\
            kernel_size=(3, 3), strides=(2, 2), activation='relu')
        # All five res blocks are the same.
        self.res_block = ResBlock(filters=128, kernel_size=(3, 3),\
            strides=(1, 1))
        # The same activation is reused several times.
        self.activation = tf.keras.layers.Activation('relu')
        # The output layer is given a custom activation function.
        self.out = tf.keras.layers.Activation(output_activation)

    def call(self, x):
        # First three convolution layers.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Five Res_block layers.
        x = self.res_block(x)
        x = self.res_block(x)
        x = self.res_block(x)
        x = self.res_block(x)
        x = self.res_block(x)
        # Two deconvolution layers.
        x = self.deconv1(x)
        x = self.deconv2(x)
        # One more convolution layer.
        x = self.conv4(x)
        # Output layer.
        return self.out(x)
