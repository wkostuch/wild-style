# Author: Michael Jurkoic

# Based on the research from this paper: https://www.mdpi.com/2071-1050/11/20/5673/htm

# All necessary Python packages are included in the requirements.txt file in 
# the root directory of the repository.
import cv2
import math
import numpy as np
import random
import unittest


def _phys_form(pix):
    """
    Converts a single pixel in a 3-channel image to three identical gray
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
def _gray_content(img):
    """
    Converts an image into grayscale.

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
        img[i] = [_phys_form(pix) for pix in img[i]]

    return img


# Zooms in on the center of an image, and returns the zoomed portion at the
# original dimensions of the input.
def _zoom_center(img, height, width):
    """
    Zooms in on the center of the given image.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    - cv2 image: centered zoom on img cropped to dimensions (height, width)
    """
    # Resize the image to twice the original size.
    large_img = cv2.resize(img, (width*2, height*2),\
        interpolation=cv2.INTER_CUBIC)

    # Crop the image to the same size as the input.
    vert_crop = height // 2
    horiz_crop = width // 2
    crop_img = large_img[vert_crop:vert_crop+height,\
        horiz_crop:horiz_crop+width, :]

    return crop_img


# _rotate_image, _largest_rotated_rect, and _crop_around_center come from
# this StackOverflow thread: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders/16778797#16778797
def _rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.array(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate)[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate)[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate)[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate)[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.array([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.array(trans_mat) * np.array(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def _crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


# Performs a random rotation on the input image.
# TODO: Fix this to return an image of the same shape as the input.
def _rand_rotate(img, height, width):
    """
    Rotates an image by a random amount.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    - cv2 image: same size as img, with blank space cropped out
    """

    # Get the rotation matrix for the required affine transform
    angle = random.randint(1, 360)
    rot_img = _rotate_image(img, angle)
    rot_crop_img = _crop_around_center(rot_img,\
        *_largest_rotated_rect(width, height, math.radians(angle)))
    # Resize the image to the shape of the input.
    rot_crop_img = cv2.resize(rot_crop_img, (width, height))

    return rot_crop_img


# Flips an image over one of its axes
def _rand_flip(img, height, width):
    """
    Flips an image over either its horizontal or vertical axis.

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
def _rand_occlude(img, height, width):
    """
    Zoom in on a corner of an image, occluding the remainder.

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

    # Resize the image to twice the original dimensions.
    large_img = cv2.resize(img, (width*2, height*2),\
        interpolation=cv2.INTER_CUBIC)
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
def _perturb(img, height, width):
    """
    Add random noise to an image.

    Parameters
    - img (cv2 image): 3-channel color image
    - height (int): height of the desired output image
    - width (int): width of the desire output image

    Returns
    cv2 image: input image with noise added
    """

    # Initialize an 'image' of the same shape as the input.
    noise = np.array([[0 for i in range(width)] for j in range(height)], )
    cv2.randn(noise, 0, 10)

    b, g, r = cv2.split(img)
    b = b + noise
    g = g + noise
    r = r + noise
    noisy_img = cv2.merge([r, g, b])

    return noisy_img


# Performs data augmentation on the style image.
def _augment_style(img, num_imgs):
    """
    Perform data augmentation on an image.

    Parameters
    - img (cv2 color image): 3-channel color image

    Returns
    - augmented_imgs (list): list of length num of random augmentations.

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

    img = img.astype("float32")

    augmentation_funcs = [_zoom_center, _rand_flip, _rand_rotate,\
        _rand_occlude, _perturb]
    # Store augmented images in here.  Length is given by num_imgs parameter.
    augmented_imgs = [None for i in range(num_imgs)]
    # Include the original image in the list.
    augmented_imgs[0] = img

    # Run the image through three randomly chosen augmentation functions, and do
    # it num_imgs times.
    for i in range(1, num_imgs):
        mut_img = img
        mut_height, mut_width = mut_img.shape[0], mut_img.shape[1]
        mut_img = augmentation_funcs[random.randint(0, 4)](mut_img, height,\
            width)
        mut_img = augmentation_funcs[random.randint(0, 4)](mut_img, mut_height,\
            mut_width)
        mut_img = augmentation_funcs[random.randint(0, 4)](mut_img, mut_height,\
            mut_width)
        augmented_imgs[i] = mut_img

    # Return a list of num_imgs augmented images.
    return augmented_imgs


# Performs preprocessing on the content image and the style image.
def augment(content, style, num_style_imgs):
    """
    Pre-processes the content and style images.

    Parameters
    - content (cv2 color image): content image
    - style (cv2 color image): style image
    - num_style_img (int): number of style image augmentations to return.

    Returns
    - (proc_content, proc_style): tuple of the processed style and content 
    images
    - proc_content is a single image
    - proc_style is a list of images

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
    proc_content = _gray_content(content)
    proc_style = _augment_style(style, num_style_imgs)

    return (proc_content, proc_style)
