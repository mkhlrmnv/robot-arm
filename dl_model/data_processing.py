"""
This file is for processing collected data
"""
import os
import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt


# Avoid OOM errors by setting GPU Memory Consumption Growth
def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')


# I moved images to test, val and train folders so images is empty,
# if data is wanted then uncomment
# But code below is more for testing and getting to know tf
"""
# gets random picture from images folder
images = tf.data.Dataset.list_files('data/images/*.jpg')
images.as_numpy_iterator().next()


# loads that picture
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


# shows is as numpy iterator, so as a lot of 3x3 matrixes
images = images.map(load_image)
images.as_numpy_iterator().next()
# print(type(images))


# batch is function that shows x amount of pictures at once
# in this case 4 next to each other
# as numpy iterator returns 3x3 matrixes of all pixels
image_generator = images.batch(4).as_numpy_iterator()


# gets next four images
plot_images = image_generator.next()


# plots four images
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)

# shows figure
# plt.show()
"""


# Below is code to move right labels to right folders
# After images were moved to test, train and val folders
# On first go I had in total 110 pictures and put 70 in train and 15 in val and test
def move_labels():
    for folder in ['train', 'test', 'val']:
        for file in os.listdir(os.path.join('data', folder, 'images')):
            filename = file.split('.')[0] + '.json'
            existing_filepath = os.path.join('data', 'labels', filename)
            if os.path.exists(existing_filepath):
                new_filepath = os.path.join('data', folder, 'labels', filename)
                os.replace(existing_filepath, new_filepath)
