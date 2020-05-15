# Author: Michael Jurkoic

import zhou_augment
import cv2
import numpy as np
import unittest

# Unit tests for zhou_augment.py
class TestAugmentFunctions(unittest.TestCase):

    # All three channels should be the same after augmentation.
    def test_gray_content(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        gray_img = zhou_augment._gray_content(img)
        b, g, r = cv2.split(gray_img)
        # Assert
        self.assertTrue((b == g).all())
        self.assertTrue((b == r).all())
        self.assertTrue((r == g).all())

    # Output should be the same size, but not identical to, the input.
    def test_zoom_center(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        zoom_img = zhou_augment._zoom_center(img, img.shape[0], img.shape[1])
        # Assert
        self.assertFalse((img == zoom_img).all())
        self.assertEqual(img.shape, zoom_img.shape)

    # Output should be same shape as input.
    def test_rand_rotate(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        rot_img = zhou_augment._rand_rotate(img, img.shape[0], img.shape[1])
        # Assert
        self.assertFalse((img == rot_img).all())
        self.assertEqual(img.shape, rot_img.shape)
    
    # Output should be the same shape, but a different image, than the input.
    def test_rand_flip(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        flip_img = zhou_augment._rand_flip(img, img.shape[0], img.shape[1])
        # Assert
        self.assertFalse((img == flip_img).all())
        self.assertEqual(img.shape, flip_img.shape)
    
    # Output should be the same size, but not identical to, the input.
    def test_rand_occlude(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        occ_img = zhou_augment._rand_occlude(img, img.shape[0], img.shape[1])
        # Assert
        self.assertFalse((img == occ_img).all())
        self.assertEqual(img.shape, occ_img.shape)

    # Output should be nearly equal to the input.
    def test_perturb(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        pert_img = zhou_augment._perturb(img, img.shape[0], img.shape[1])
        # Assert
        self.assertFalse((img == pert_img).all())
        self.assertEqual(img.shape, pert_img.shape)

    # Check to see the _augment_style returns the correct number of images.
    def test_augment_style(self):
        # Arrange
        img = cv2.imread("../images/test/test_kitten.png", cv2.IMREAD_COLOR)
        # Act
        augments = zhou_augment._augment_style(img, 10)
        # Assert
        self.assertEqual(len(augments), 10)


def test():
    unittest.main()

    
if __name__ == "__main__":
    test()