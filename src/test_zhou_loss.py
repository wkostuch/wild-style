# Author: Michael Jurkoic

import cv2
import methods as meth
import numpy as np
import tensorflow as tf
import unittest
import zhou_loss as loss


# Unit tests for zhou_loss.py
class TestLossFunctions(unittest.TestCase):

    # Make sure content loss is working.  The content loss between two identical
    # images should be 0.
    def test_content_loss(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        img_tensor = meth.cv2_image_to_tensor(img)
        # Act
        content_loss = loss._content_loss(img_tensor, img_tensor)
        # Assert
        self.assertEqual(content_loss, 0)

    # Check to see if the Gram matrix calculation is returning a tensor of the
    # right size.  If c is the number of channels of the input, the output
    # should be a c x c x 1 tensor.
    def test_gram_matrix(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        img_tensor = meth.cv2_image_to_tensor(img)
        # Act
        gram = loss._gram_matrix(img_tensor)
        shape = tf.shape(gram).numpy()
        # Assert
        self.assertEqual(shape[0], 3)
        self.assertEqual(shape[1], 3)


def test():
    unittest.main()
 
if __name__ == "__main__":
    test()
