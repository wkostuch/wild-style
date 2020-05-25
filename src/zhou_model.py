# Author: Michael Jurkoic

# Based on this paper by Zhou, et. al.: 
# https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import methods as meth
import os
import random
import tensorflow as tf
import zhou_augment
import zhou_loss
from zhou_stylize import Stylizer


class ZhouModel:
    """
    A class for building, training, and exporting a model for applying a given
    style to any content image.  Uses a custom CNN for stylizing images, and
    uses VGG19 for training.

    Constructor parameters
    - content_img_path (string): Path to the content image.
    - style_img_path (string): Path to the style image.
    - content_layers (list of strings): List of VGG19 layers used in activating
    the content image.  Defaults to Conv3_1
    - style_layers (list of strings): List of VGG19 layers used in activating
    the style image.  Defaults to [Conv2_1, Conv3_1, Conv4_1, Conv5_1].
    - output_activation (string): The activation function to be used on the
    output layer of the stylization network.
    - num_augmentations (int): Number of augmentations of the style image to be
    produced.  Defaults to 5000.
    - img_size (int): Size in pixels of the model output image.  All outputs are
    square images of img_size x img_size.
    """


    # Class constructor.
    def __init__(self, content_img_path, style_img_path,\
        content_layers=['block3_conv1'],\
        style_layers=['block2_conv1', 'block3_conv1', 'block4_conv1',\
        'block5_conv1'], output_activation='sigmoid', num_augmentations=5000,\
        img_size=256):

        # Instantiate an untrained stylization model.
        self.zhou = Stylizer(output_activation=output_activation)

        # Set up two VGG19 networks; one for activating the content image, one
        # for activating the style image.
        self.vgg_content = self.__init_vgg(content_layers)
        self.vgg_style = self.__init_vgg(style_layers)

        # Create fields for the content, style, and stylized images.
        self.content_img = self.__load_img(content_img_path)
        self.style_img = self.__load_img(style_img_path)
        self.stylized_img = None

        # Get the augmented inputs.
        self.content_augment, self.style_augments\
            = self.__augment_data(content_img_path, style_img_path,\
            num_augmentations, img_size)


    # A function for setting up the VGG19 instance.
    def __init_vgg(self, layers):
        """
        Sets up an instance of VGG19 to return the output from the desired
        content and style layers.

        Parameters
        - layers (list of strings): List of desired output layers.

        Returns
        - model (Keras model): VGG19 model with the specified content and style
        layer outputs.
        """

        # Initialize the VGG19 model.
        vgg19 = tf.keras.applications.VGG19(include_top=False,\
            weights='imagenet')
        vgg19.trainable = false

        # Set up and return the model with outputs from the specified layers.
        outputs = [vgg19.get_layer(name) for name in layers]
        model = tf.keras.Model([vgg.input], outputs)
        return model


    # Uses the zhou_augment module to get augmented images.
    def __augment_data(self, content_img_path, style_img_path,\
        num_augmentations, img_size):
        """
        Obtain augmentations of the training data.

        Parameters
        - content_img_path (string): Path to the content image.
        - style_img_path (string): Path to the style image.
        - num_augmentations (int): Number of augmentations of the style image to
        be returned.
        - img_size (int): Size in pixels of the image to be returned (returned
        images are square).

        Returns
        - augmentations (tuple): Tuple consisting of the augmented content image
        and a list of the augmented style images.
        """

        # Get tensor representations of the content and style images, and
        # squarify them.
        content_tensor = meth.load_img_as_tensor(content_img_path,\
            dimension=img_size)
        square_content_tensor = meth.squarify_tensor(content_tensor)
        style_tensor = meth.load_img_as_tensor(style_img_path,\
            dimension=img_size)
        square_style_tensor = meth.squarify_tensor(style_tensor)

        # Get the augmented images as a tuple
        augment_tuple = zhou_augment.augment(square_content_tensor,\
            square_style_tensor, num_augmentations)
        return augment_tuple


    def __total_loss(self, style_img, loss_weights):
        """
        Computes total loss using the functions in the zhou_loss model.

        Parameters
        - style_img (image tensor): An augmented style image.
        - loss_weights

        Returns
        - loss (float): The total loss computed from the stylized image.

        The VGG19 layers of the loss network come into play here.
        """

        # Activate the content image with the VGG19 network, and activate the
        # synthesized image with the same VGG layers.
        activated_content = self.vgg_content(self.content_augment)
        content_activated_synth = self.vgg_content(self.stylized_img)
        # Activate the style image with the VGG19 network, and activate the 
        # synthesized image with the same VGG layers.
        activated_styles = self.vgg_style(style_img)
        style_activated_synths = self.vgg_style(self.stylized_img)

        # Use the zhou_loss module to compute the total loss.
        loss = zhou_loss.total_loss(activated_content, content_activated_synth,\
            activated_styles, style_activated_synths, self.stylized_img,\
            loss_weights)

        return loss


    def __content_loss(self):
        """
        Computes content loss using functions in the zhou_loss module.

        Returns
        - loss (float): The computed content loss within the stylized image.

        This function uses the un-augmented content image.
        """

        # Activate the content image and the stylized image using the same VGG19
        # layers.
        activated_content = self.vgg_content(self.content_img)
        activated_synth = self.vgg_content(self.stylized_img)

        # Compute the content loss.
        loss = zhou_loss.content_loss(activated_content, activated_synth)

        return loss


    @tf.function
    def __train_step(self, style_ref, total_loss, opt,\
        loss_weights=(1.0, 1.0, 1.0)):
        """
        Performs a single training step, stylizing the image, computing the
        loss, and updating the stylization network weights.

        Parameters
        - style_ref (image tensor): The style image augmentation to be used in 
        calculating the style loss.
        - total_loss (Boolean): A Boolean variable that determines which loss
        function to use.  If True, use the total loss function.  If false, use
        the content loss function.
        - opt (TensorFlow Keras optimizer): The optimizer to be used in updating
        the gradient.
        - loss_weights (optional tuple of floats): Weights for computing the 
        weighted total loss.  Defaults to (1.0, 1.0, 1.0).

        No return value.  Updates the variables of the Zhou model contained
        within the object.
        """

        with tf.GradientTape as tape:
            # Stylize the image using the stylization model.
            self.stylized_img = self.zhou(self.content_img)
            # Compute the loss.
            # TODO: Add loss computation wrappers to the class.
            loss = None
            if total_loss == True:
                loss = self.__total_loss(style_ref, loss_weights)
            else:
                loss = self.__content_loss()

        grads = tape.gradient(loss, self.zhou.trainable_variables)
        opt.apply_gradients(zip(grads, self.zhou.trainable_variables))


    # Train the model.
    def train(self, loss_weights=(1.0, 5.0, 1.0), epochs=100, batch_size=4,\
        learning_rate=1e-3):
        """
        Trains the stylization model.

        Parameters
        - loss_weights (tuple of floats): Weights for the weighted loss 
        function.  The tuple takes the form (alpha, beta, gamma) where alpha is
        the content loss weight, beta is the style loss weight, and gamma is the
        total variance loss weight.  Defaults to (1.0, 5.0, 1.0)
        - epochs (int): Number of training epochs.  Defaults to 100.
        - batch_size (int): Batch size of the training data.  Defaults to 4.
        - learning_rate (float): Model learning rate.  Defaults to 0.001.

        Returns
        - model (Keras model): Trained Keras model.

        The model is saved as the ZhouModel object's state, so calling this
        function repeatedly will continue to train the already trained model.
        """

        # Initialize the optimizer.
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Initialize the loss function flag.
        use_total_loss = True

        # Iterate over the given number of epochs.
        for epoch in range(epochs):
            print(f"Beginning Epoch {epoch}/{epochs}")
            # Get the number of iterations, then iterate over them.
            iterations = len(self.style_augments) // batch_size
            for i in range(iterations):
                # Use total loss function on even iterations.  Use content loss
                # only on odd iterations.
                if i % 2 == 0:
                    use_total_loss = True
                else:
                    use_total_loss = False
                # Get style image augmentations, working through all the style 
                # augmentations consecutively.
                batch = []
                for j in range(batch_size):
                    style_datum = self.style_augments[random.randint(0,\
                        len(self.style_augments))]
                    batch.append(style_datum)
                for img in batch:
                    self.__train_step(img, use_total_loss, opt)

            # Display the stylized image at the end of each epoch.
            # TODO: Delete this line once testing is completed.
            meth.display_tensor_as_image(self.stylized_img)


    # TODO: Add functionality for saving models and weights.
    # TODO: Add functionality for loading saved weights into the model.
    # TODO: Add functionaliyt for displaying the stylized image.


def test():
    # Initialize some variables for testing
    content_path = ""
    style_path = ""
    weights = (1.0, 1.0, 1.0)
    epochs = 5

    # Build the model class and do some training.
    model = ZhouModel(content_path, style_path)
    model.train(loss_weights=weights, epochs=epochs)


if __name__ == "__main__":
    test()
