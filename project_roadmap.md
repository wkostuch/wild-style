


* For points to be done, ~ for answers to the points

# Roadmap for the project

## Finished
* Implemented basic style transfer algorithm from TensorFlow tutorial
   * Done.  Initial demo based on tutorial here: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb#scrollTo=dqB6aNTLNVkK

## Next Up
* Set up Python virtual environment for the code base.
* Discover new wasy of creating output images.
   * Try pulling different layers from the VGG19 model.
   * Try giving different weights to content/style in the loss function.
   * Different training processes.
   * More photrealistic vs more painterly.

## Down the Road
* Compile some pretrained models for "plug and play" functionality.
* Do style transfer in real-time with video.
* Do style transfer on specific objects in the image, while preserving the rest of the image.
* Unify under a single user interface (graphical or command line)


# Android App Feature List

* Slider-type thing for style vs content 
    * Photo-realstic on one end, style distortion on the other end

* Pick two images from camera roll
    * Have some style references built in
    * Camera support? 
        * If it's fast enough: video with limited frames; if it's slow then perhaps time-lapse 
