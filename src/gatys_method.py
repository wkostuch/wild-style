'''
This code is based upon the code found at the TensorFlow website
for neural style transfer: https://www.tensorflow.org/tutorials/generative/style_transfer
It has been expanded and reworked for our own purposes.
'''

'''
Set up all the necessary imports 
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import functools
import cv2
import methods as meth
'''
End of imports
'''


'''
Class for getting the style and content layers from
the VGG19 network.
'''
class StyleContentModel(tf.keras.models.Model):
    """A class for getting the content and style layers from the VGG19 network."""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        # Scale inputs up
        inputs = inputs*255.0
        # Preprocess the input 
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        # Get the content and style layers
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])
        # Run the style layers through the Gram matrix to get the style of the image
        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]
        # Build a dict for the content layers
        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        # Build a dict for the style layers
        style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
        # Return a dict mapping content and style to their respective
        # dictionaries built above
        return {'content':content_dict, 'style':style_dict}

'''
End of StyleContentModel class
'''

'''
Some useful methods
'''
def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load the VGG19 model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Pull the layers from the VGG model and put them in a list
    outputs = [vgg.get_layer(name).output for name in layer_names]
    # Make a new model with those layers that lets us grab 
    # values for a particular layer in the VGG19 network
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Gram matrix for calculating the style."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def clip_0_1(image):
    """Clips the values in a tensor to be between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

'''
End of useful methods
'''


'''
Class for actually doing the neural style transfer in the Gatys method
'''
class GatysNeuralStyleTransfer():
    """A class containing everything required for neural style transfer
    using the Gatys method.
    
    Constructor takes a:
    - path_to_content_image (str): filepath to the image used for its content
    - path_to_style_image (str): filepath to the image used for its style
    - content_layers (list): list of layers to pull from VGG19 for content (defaults given)
    - style_layers (list) : list of layers to pull from VGG19 for style (defaults given)
    - content_weight (int) : how much to weight the content (default given)
    - style_weight (int) : how much to weight the style (default given)
    - optimizer (tf.optimizer) : optimizer for the style transferral 
    
    Supported file types for content/style images:
    - .png"""

    # Initizalize some fields for the class
    # Tensor for the content image
    content_image_tensor: None 
    # Tensor for the style image
    style_image_tensor: None 
    # Tensor for the "image so far" while doing the style transfer 
    transfer_image_tensor: None 
    # Pull up the VGG19 model
    VGG_model = None
    # List of layers for content
    content_layers = None
    # List of layers for style
    style_layers = None
    # Content and style weights
    content_weight = None
    style_weight = None
    # Make a StyleContentModel 
    extractor = None
    # Weight for the variation in styling 
    total_variation_weight = None
    # Optimizer 
    optimizer = None
    # Targets for content and style
    style_targets = None
    content_targets = None
    # Tracker for how many times the transfer_image has been styled
    style_transfer_steps = None

    # Nice little constructor for the class
    def __init__(self, path_to_content_image: str, path_to_style_image: str, 
                    content_layers=['block5_conv2'],  
                    style_layers=['block1_conv1', 'block2_conv1', 
                                'block3_conv1', 'block4_conv1',  
                                'block5_conv1'],
                    content_weight=1e2, style_weight=1e-3,
                    total_variation_weight=30,
                    optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)):
        # Create two tensors from the images given for content and style
        self.content_image_tensor = meth.load_img_as_tensor(path_to_content_image)
        self.style_image_tensor = meth.load_img_as_tensor(path_to_style_image)
        # Nothing's been done to style, so the stylized image is the content image
        self.transfer_image_tensor = tf.Variable(self.content_image_tensor)
        # Assign the VGG19 model to our field
        self.VGG_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        # Assign content and style layers appropriately 
        self.content_layers = content_layers
        self.style_layers = style_layers
        # Assign weights for content and style
        self.content_weight = content_weight
        self.style_weight = style_weight
        # Make an extractor via StyleContentModel 
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        # Set the variation weight
        self.total_variation_weight = total_variation_weight
        # Set the optimizer
        self.optimizer = optimizer
        # Update the content and style targets
        self.content_targets = self.extractor(self.content_image_tensor)['content']
        self.style_targets = self.extractor(self.style_image_tensor)['style']
        # Set up the step tracker
        self.style_transfer_steps = 0

    def show_content_image(self):
        """Displays the content image using the system's default photo viewer."""
        meth.display_tensor_as_image(self.content_image_tensor)
    
    def show_style_image(self):
        """Displays the style image using the system's default photo viewer."""
        meth.display_tensor_as_image(self.style_image_tensor)

    @tf.function()
    def style_step(self):
        """Styles the image one increment."""
        image = self.transfer_image_tensor
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))
    
    def style_n_times(self, num_times_to_style: int):
        """Runs style(step) on the transfer_image num_times_to_style times."""
        for i in range(num_times_to_style):
            self.style_step()
        self.style_transfer_steps += num_times_to_style
        

    def style_content_loss(self, outputs):
        """Returns the total loss for style transferral."""
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(self.style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(self.style_layers)
        loss = style_loss + content_loss
        return loss

    def show_currently_styled_image(self):
        """Displays the styled image via the system's default photoviewer."""
        meth.display_tensor_as_image(self.transfer_image_tensor)
        

'''
End of GatysNeuralStyleTransfer class
'''



content_path = "../images/content/lakeside.png"
style_path =  "../images/style/parable_of_sower.png"
transfer = GatysNeuralStyleTransfer(content_path, style_path)
transfer.style_n_times(10)
transfer.show_content_image()
transfer.show_style_image()
transfer.show_currently_styled_image()
print(transfer.style_transfer_steps)

        