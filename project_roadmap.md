


* For points to be done, ~ for answers to the points

# Roadmap for the project

* Finish style transfer algorithm from TensorFlow website
    * Play around with it and see what we can build on, what we want to change
        * Pull from different layers, see what we get
        * Set up a fat script to run through lots of permutation and see what we get as results

* Slider with style vs content 

* Look into "plug and play" with models built in Python transferring to Kotlin/other languages

* VGG19 finds objects, we then style those and put them back into the image

* Need to look into size mis-match between pictures
    * Aspect ratio, then upscale/downscale






# App Feature List

* Slider-type thing for style vs content 
    * Photo-realstic on one end, style distortion on the other end

* Pick two images from camera roll
    * Have some style references built in
    * Camera support? 
        * If it's fast enough: video with limited frames; if it's slow then perhaps time-lapse 
        
* Pre-built models, kind of like picking a filter but instead it's a model 


* Bonus: 
    * Object detection where you can overlay a styled version of that (say take a tree, style it, then put it back into the photo)
