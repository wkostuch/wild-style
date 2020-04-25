# Based on the research from this paper: https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import cv2
import method_library as meth
import numpy as np
import tensorflow as tf
from PIL import Image
import random


def phys_form(pix):
    """Converts a single pixel in a 3-channel image to three identical gray
    channels.

    Parameters
    - pix (1x3 array): single 3-channel pixel

    Returns
    - 1x3 array

    Uses the BGR conversion formula: Gray(x) = B(x) * 0.144 + G(x) * 0.587 + 
    R(x) * 0.299, and gives all three channels identical values.
    """

    # Calculate the gray pixel value.
    blue_weighted_val = pix[0] * 0.144
    green_weighted_val = pix[1] * 0.587
    red_weighted_val = pix[2] * 0.299
    gray_val = blue_weighted_val + green_weighted_val + red_weighted_val

    # Assign the gray pixel value to all three channels.
    new_pix = [gray_val, gray_val, gray_val]

    return new_pix


# Takes the content image and turns it into a gray image.
# Graying the content image helps prevent the color mixing that occurs in the
# classic Gatys style transfer algorithm.
def gray_content(img):
    """Converts an image into grayscale.

    Parameters
    - img (cv2 image): 3-channel color image

    Returns
    - grayscale image

    Converts each pixel in the input image into grayscale by the formula: 
    Gray(x) = R(x) * 0.299 + G(x) * 0.587 + B(x) * 0.144, where Gray(x) is a
    pixel in the resulting image, and R(x), G(x), and B(x) are pixels in the
    red, green, and blue color channels, respectively.

    The resulting image should be treated as grayscale, but it retains three
    channels to match the 3-channel format of the style image.
    """

    # Cycle through every pixel in the image, and replace all three channels
    # with their grayscale equivalents.
    for i in range(len(img)):
        img[i] = [phys_form(pix) for pix in img[i]]

    return img


# Zooms in on the center of an image, and returns the zoomed portion at the
# original dimensions of the input.
def zoom_center(img, height, width):
    """Zooms in on the center of the given image.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    - cv2 image: centered zoom on img cropped to dimensions (height, width)
    """
    # Upsample the image to double the size of the original.
    # This is effectively x2 zoom.
    large_img = cv2.pyrUp(img)

    # Crop the image to the same size as the input.
    vert_crop = height // 2
    horiz_crop = width // 2
    crop_img = large_img[vert_crop:vert_crop+height,\
        horiz_crop:horiz_crop+width, :]

    return crop_img


# Performs a random rotation on the input image.
def rand_rotate(img, height, width):
    """Rotates an image by a random amount.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    - cv2 image: same size as img, with blank space cropped out
    """

    # Get the rotation matrix for the required affine transform
    center = (height // 2, width // 2)
    angle = random.randint(1, 360)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    rot_img = cv2.warpAffine(img, rot_mat, (height, width))

    # Zoom in on the center of the image to hopefully crop out unwanted blank
    # space left on the edges after rotation.
    # TODO: Find a better way to get rid of blank space after rotation.
    rot_img = zoom_center(img, height, width)

    return rot_img


# Flips an image over one of its axes
def rand_flip(img):
    """Flips an image over either its horizontal or vertical axis.

    Parameters
    - img (cv2 image): 3-channel color image

    Returns
    - cv2 image: original image flipped over one of its axes

    Randomly determines which axis (vertical, horizontal or both) to flip the 
    image over.
    """

    # Decide which axis to flip the image over, then perform the flip.
    axis_num = random.randint(-1, 2)
    flip_img = cv2.flip(img, axis_num)

    return flip_img


# Randomly occludes part of an image.
# Much like zoom_center, but zooms in on a corner of the image rather than the
# center.
# TODO: See if rand_occlude and zoom_center can be merged.
def rand_occlude(img, height, width):
    """Zoom in on a corner of an image, occluding the remainder.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    - cv2 image: zoomed in on one of the four corners

    The corner to zoom in on is picked at random.
    """

    # Pick a random corner to zoom in on.
    # 1 = top left, 2 = top right, 3 = bottom right, 4 = bottom left
    rand_corner = random.randint(1, 5)

    # Upsample the image to double the size of the original.
    # This is effectively x2 zoom.
    large_img = cv2.pyrUp(img)

    large_dims = large_img.shape
    
    # Slice out the required part of the image
    if rand_corner == 1:
        crop_img = large_img[:height, :width, :]
    elif rand_corner == 2:
        crop_img = large_img[:height, large_dims[1]-width:, :]
    elif rand_corner == 3:
        crop_img = large_img[large_dims[0]-height:, large_dims[1]-width:, :]
    elif rand_corner == 4:
        crop_img = large_img[large_dims[0]-height:, :width, :]
    else:
        crop_img = large_img[:height, :width, :]
    
    return crop_img


# Adds random noise to an image
def perturb(img, height, width):
    """Add random noise to an image.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    cv2 image: input image with noise added
    """

    # Initialize an 'image' of the same shape as the input.
    noise = [[0 for i in range(width)] for j in range(height)]
    cv2.randn(noise, 0, 0.1)

    noisy_img = img + noise

    return noisy_img


# Performs data augmentation on the style image.
# TODO: consider adding more rotations to the output.
def augment_style(img):
    """Perform data augmentation on an image.

    Parameters
    - img (cv2 color image): 3-channel color image

    Returns
    - tuple: 6-tuple containing the augmented images

    Perform data augmentation on the input image with a series of image
    transformations.  These transformations include zooming in on the original
    image, randomly rotating the image, flipping the image horizontally or 
    vertically, randomly occluding part of the image, and randomly perturbing 
    the image RGB values with Gaussian noise.
    """

    # Get the dimensions of the original image for reference.
    dims = img.shape
    height = dims[0]
    width = dims[1]

    # Get a separate image for each augmentation transformation.
    zoom_img = zoom_center(img, height, width)
    rot_img = rand_rotate(img, height, width)
    flip_img = rand_flip(img)
    occlude_img = rand_occlude(img)
    perturb_img = perturb(img, height, width)

    # Return a tuple of all 5 augmented images and the original.
    return (img, zoom_img, rot_img, flip_img, occlude_img, perturb_img)


# Performs preprocessing on the content image and the style image.
def preprocess(content, style):
    """Pre-processes the content and style images.

    Parameters
    - content (cv2 color image): content image
    - style (cv2 color image): style image

    Returns
    - (proc_content, proc_style): tuple of the processed style and content 
    images

    The content image is preprocessed by converting the image to grayscale, then
    duplicating that grayscale image across all three image channels, ensuring
    the processed content image has the same dimensions as the style image, but
    the problem of color mixing present in the Gatys style transfer is avoided.

    The style image is processed by subjecting it to a number of transformations
    in order to augment the texture and color data and reduce the impact of
    certain texture or color patterns being concentrated in one part of the
    image.
    """

    # Conver the content image to 3 channels of grayscale.
    proc_content = gray_content(content)


def main():
    return


if __name__ == "__main__":
    main()
