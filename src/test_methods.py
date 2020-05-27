# Author: Michael Jurkoic

import cv2
import methods as meth
import numpy as np
import tensorflow as tf
import unittest


# Unit tests for methods.py
class TestMethods(unittest.TestCase):

    # Make sure squarify_image works.  The returned image should have equal
    # width and height.
    def test_squarify_img(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        dim = 256
        # Act
        square_image = meth.squarify_image(img, dim)
        # Assert
        self.assertEqual(square_image.shape[0], square_image.shape[1])
        self.assertEqual(square_image.shape[2], 3)
        self.assertEqual(square_image.shape[0], dim)
        self.assertEqual(square_image.shape[1], dim)


def test():
    unittest.main()

if __name__ == "__main__":
    test()