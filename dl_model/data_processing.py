"""
This file is for processing collected data
"""

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

# Avoid OOM errors by setting GPU Memory Consumption Growth
""" 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
"""

images = tf.data.Dataset.list_files('data/images/*.jpg')
images.as_numpy_iterator().next()


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


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
plt.show()
