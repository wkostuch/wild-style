# Author: Michael Jurkoic

# Data pipeline for training the ZhouModel class.

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import methods as meth
import random
import tensorflow as tf
import zhou_augment


class Pipeline:
    """
    A class for handling the data pipeline for training ZhouModel instances.  It
    is an iterable object, so iterating over it will give a batch of training
    data augmentations.

    Constructor parameters
    - style_image (cv2 image): Style image.  All training data will be
    augmentations of this image.
    """


    # Class constructor.
    def __init__(self, style_image):
        self.source = style_image
    

    # Calling next() on the iterator returns the next batch of augmentations as
    # a list of tensors.
    def get_augmentation(self):
        """
        Return a style image augmentation.

        Returns
        - data (image tensor): Augmented image.
        """

        # Get a single image augmentation.
        augment = zhou_augment._augment_style(self.source, 1)
        # Turn the augmentation into a tensor.
        data = meth.cv2_image_to_tensor(img)
        # Return a batch of image tensors.
        return data
