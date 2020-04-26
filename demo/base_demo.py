# The following code is based off of the tutorial from:
# https://www.tensorflow.org/tutorials/generative/style_transfer
# and modified and expanded upon as needed for our project.  


'''
Set up all the necessary imports 
'''
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from PIL import Image

import time

import functools

import cv2
'''
End of imports
'''



#Get the file-paths for our images
content_path = "../images/content/rome_waterfront.png"
#style_path =  "../images/style/monet_haystacks.png"
style_path = "../images/results/rome+starry_night.png"

#Load the image into a tensor
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #Compute new dimensions for the image
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    #Resize the image-tensor
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img



#Takes a tensor and returns one that can be displayed via Pillow
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


#Takes an image-tensor, makes it displayable, and displays it
def imshow(image, title=None):
    if len(image.shape) >= 3:
        image = tf.squeeze(image, axis=0)
    img = tensor_to_image(image)
    if title:
        img.show(title=title)
    else: 
        img.show(title="Picture")


content_image = load_img(content_path)
style_image = load_img(style_path)

imshow(content_image, title='Content Image')

imshow(style_image, title='Style Image')


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
  print(layer.name)


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


style_weight=1e-2
content_weight=1e4


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

total_variation_weight=30

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


#train_step(image)
#train_step(image)
#train_step(image)
'''
Block for styling the picture
'''

image = tf.Variable(content_image)

start = time.time()

epochs = 20
steps_per_epoch = 10
total_steps = epochs * steps_per_epoch

#Run the picture through the styling method a bunch of times
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  #display.clear_output(wait=True)
  print("Train step: {} out of {}".format(step, total_steps))

end = time.time()
print("Total time: {:.1f}".format(end-start))


#Yo, want to see that sweet pic?  Hell yeah!
img = tensor_to_image(image)
img.show(title="pic") 
'''
Styling is over!
'''

'''
Ask if the user wants to save the picture
'''
save_answers = ["y", "Y", "Yes", "yes", "ye", "Ye"]
do_not_save_answers = ["n", "N", "No", "no"]

file_path = "../images/results/"
file_name = file_path + "rome+haystacks_150_steps.png"
while(True):
    answer = input("Would you like to save this picture? (y/n): ")
    # Case for: invalid input.
    if answer not in save_answers and answer not in do_not_save_answers:
        print("Hmm, that didn't look like a valid input...please try again.")
        continue
    # Case for: the user doesn't want to save the picture.
    elif answer in do_not_save_answers:
        print("Picture not being saved.  Hopefully you'll find one you like better in the future.")
        break
    # Case for: the user does want to save the picture.
    elif answer in save_answers:
        print("Your lovely creation is being saved, please wait a moment. . .")
        img.save(file_name, "PNG")
        print("Your picture has been saved at: " + file_name)
        break
'''
End of saving
'''

