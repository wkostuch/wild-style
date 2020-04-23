# Based on the research from this paper: https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import cv2
import method_library as meth
import numpy as np
import tensorflow as tf
from PIL import Image


# Takes the content image and turns it into a gray image.
# Graying the content image helps prevent the color mixing that occurs in the
# classic Gatys style transfer algorithm.
def gray_content(img):
    """Converts an image into grayscale.

    Parameters:
    img (cv2 image): color image

    Returns:
    grayscale image

    Converts each pixel in the input image into grayscale by the formula: 
    Gray(x) = R(x) * 0.299 + G(x) * 0.587 + B(x) * 0.144, where Gray(x) is a
    pixel in the resulting image, and R(x), G(x), and B(x) are pixels in the
    red, green, and blue color channels, respectively.
    """


def main():
    return


if __name__ == "__main__":
    main()
