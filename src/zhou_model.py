# Author: Michael Jurkoic

# Based on this paper by Zhou, et. al.: 
# https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import cv2
import methods as meth
import random
import tensorflow as tf
import zhou_augment
import zhou_loss
from zhou_pipeline import Pipeline
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

        # Save the image filepaths.
        self.content_path = content_img_path
        self.style_path = style_img_path

        # Save the image size for easy access.
        self.img_size = img_size

        # Set up two VGG19 networks; one for activating the content image, one
        # for activating the style image.
        self.vgg_content = self.__init_vgg(content_layers)
        self.vgg_style = self.__init_vgg(style_layers)

        # Create fields for the content, style, and stylized images.
        self.content_img = self.__load_image(content_img_path)
        self.style_img = self.__load_image(style_img_path)
        # Initialize the stylized image to just be the content image.  The image
        # stored in this field will be updated as the model is trained.
        self.stylized_img = self.__load_image(content_img_path)

        # Get the augmented content.
        # Augmented style data will be provided as needed during training.
        self.content_augment = self.__augment_content(content_img_path)


    # Custom handler function for loading images, because the class currently
    # only works with square images.
    # TODO: Generalize to various sizes and aspect ratios.
    def __load_image(self, img_path):
        """
        Loads an image from the given path into a square tensor with sides of
        the given dimension.

        Parameters
        - img_path (string): Path to the image.

        Returns
        - square_image_tensor (tensor)
        """

        # Load the image with OpenCV, and squarify it.
        image = meth.import_image(img_path)
        square_image = meth.squarify_image(image, self.img_size)
        square_image_tensor = meth.cv2_image_to_tensor(square_image)

        return square_image_tensor


    # A function for obtaining the augmentation of the content image.
    def __augment_content(self, img_path):
        """
        Loads an image from the given path into a square image, and returns a
        tensor representation of its content augmentation.

        Parameters
        - img_path (string): Path to the image.

        Returns
        - img_tensor (tensor)
        """

        # Load the image with OpenCV, and squarify it.
        img = meth.import_image(img_path)
        square_image = meth.squarify_image(img, self.img_size)
        # Get the content augmentation and convert it to a tensor.
        zhou_augment._gray_content(img)
        img_tensor = meth.cv2_image_to_tensor(img)

        return img_tensor


    # Sets up the training pipeline.
    def __build_training_pipeline(self, style_img_path, batch_size):
        """
        Sets up and returns a training pipeline.

        Parameters
        - style_img_path (string): Path to the file location of the style image.
        - batch_size (int): Number of augmentations to be contained in a batch.

        Returns
        - pipeline (iterable object): Iterable object that produces batches of
        style image augmentations.
        """

        # Load the image with OpenCV, and squarify it.
        img = meth.import_image(img_path)
        square_image = meth.squarify_image(img, self.img_size)
        # Get the data pipeline.
        pipeline = Pipeline(square_image)

        return pipeline


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
        vgg19.trainable = False

        # Set up and return the model with outputs from the specified layers.
        outputs = [vgg19.get_layer(name).output for name in layers]
        model = tf.keras.Model([vgg19.input], outputs)
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

        # Get OpenCV representations of the content and style images for 
        # augmentation.
        con_img = meth.import_image(content_img_path)
        con_img = meth.squarify_image(con_img, img_size)
        sty_img = meth.import_image(style_img_path)
        sty_img = meth.squarify_image(sty_img, img_size)

        # Get the augmented images as a tuple
        augment_tuple = zhou_augment.augment(con_img, sty_img,\
            num_augmentations)

        # Turn the images in the tuple into tensors.
        proc_content = meth.cv2_image_to_tensor(augment_tuple[0])
        proc_styles = []
        # For testing
        i = 0
        for image in augment_tuple[1]:
            proc_styles.append(meth.cv2_image_to_tensor(image))
            i += 1
            print(f"{i} style image augmentations produced")

        return (proc_content, proc_style)


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
    # TODO: Figure out how to do batching.
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
        # Initialize the training data pipeline.
        pipe = self.__build_training_pipeline(self.style_path)
        # Get a TensorFlow Dataset object
        dataset = tf.data.Dataset.from_generator(pipe.get_augmentation,\
            tf.float32)
        # Get the dataset in batches.
        dataset = dataset.batch(batch_size).repeat(epochs)

        # Iterate over the given number of epochs.
        for epoch in range(epochs):
            print(f"Beginning Epoch {epoch}/{epochs}")
            ds_iter = iter(dataset)
            for batch in ds_iter:
                # Use total loss function on even iterations.  Use content loss
                # only on odd iterations.
                if i % 2 == 0:
                    use_total_loss = True
                else:
                    use_total_loss = False
                # Train on style image augmentations.
                self.__train_step(batch, use_total_loss, opt)

            # Display the stylized image at the end of each epoch.
            # TODO: Delete this line once testing is completed.
            meth.display_tensor_as_image(self.stylized_img)


    # TODO: Add functionality for saving models and weights.
    # TODO: Add functionality for loading saved weights into the model.
    # TODO: Add functionality for exporting TensorFlow Lite models.
    # TODO: Add functionality for displaying the stylized image.


def test():
    # Initialize some variables for testing
    content_path = "../images/content/dubrovnik.png"
    style_path = "../images/style/starry_night.png"
    weights = (1.0, 1.0, 1.0)
    epochs = 5

    # Build the model class and do some training.
    model = ZhouModel(content_path, style_path)
    model.train(loss_weights=weights, epochs=epochs)


if __name__ == "__main__":
    test()
