


* For points to be done, ~ for answers to the points

# Roadmap for the project

## Finished
* Implemented basic style transfer algorithm from TensorFlow tutorial
  ~ Initial demo based on tutorial here: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb#scrollTo=dqB6aNTLNVkK
* Set up Python virtual environment for the code base.
  ~ Added requirements.txt file and README instructions for setting up the required dependencies.
* Refactored Gatys method into a class in gatys_method.py

## Next Up
* Discover new wasy of creating output images.
  * Try pulling different layers from the VGG19 model.
  * Try giving different weights to content/style in the loss function.
  * Different training processes.
  * More photrealistic vs more painterly.
* Fold pretrained more closely into the codebase.
  * Eliminate the need for dowloading VGG19 weights before running the code.
  * Get a locally-based copy of the model for fast style transfer.
* Set up a nice data pipeline for the image augmentations.
   

## Down the Road
* Compile some pretrained models for "plug and play" functionality.
* Do style transfer in real-time with video.
* Do style transfer on specific objects in the image, while preserving the rest of the image.
* Unify under a single user interface (graphical or command line)
* Experiment with separating foreground and background styles to achieve a better results with Renaissance styles.
  * This model might be helpful: https://github.com/NathanUA/U-2-Net
  * Credit to Michael Booton for the suggestion.


# Android App Feature List

* Slider-type thing for style vs content 
  * Photo-realstic on one end, style distortion on the other end

* Pick two images from camera roll
  * Have some style references built in
  * Camera support? 
    * If it's fast enough: video with limited frames; if it's slow then perhaps time-lapse 
