"""
This code is for albumentations
Basically for showing right labels on top of right picture,
also to make a lot more data by shuffling it and changing pictures parameters as example brightness
"""

import albumentations as alb
import os
import cv2
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from data_processing import load_image, limit_gpu

# parameters by whom, program will be changing pictures and
# Original tutorial had cropping "alb.RandomCrop(width=450, height=450),", but I wanted to drop it
# to get full screen detection
augmentor = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)],
    bbox_params=alb.BboxParams(format='albumentations',
                               label_fields=['class_labels']))

# code block to just test albumentations, and how to draw label on the picture
"""
# takes some picture and label and prints out max and mix coordinates of label
# prints out [[x_min, y_min], [x_max, y_max]]
img = cv2.imread(os.path.join('data', 'train', 'images', 'b80ac550-a0d2-11ee-b134-aa0f398c44fa.jpg'))

# prints out data of image
# cv2.imshow("image", img)
# print(img.shape)

with open(os.path.join('data', 'train', 'labels', 'b80ac550-a0d2-11ee-b134-aa0f398c44fa.json'), 'r') as f:
    label = json.load(f)
# print(label['shapes'][0]['points'])

# puts label coordinates into one array, so [x_min, y_min, x_max, y_max]
coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]
# print(coords)

# divides coordinates by wight and height on image
# more useful when images are getting cropped
coords = list(np.divide(coords, [1920, 1080, 1920, 1080]))
# print(coords)

# for testing augmenations
augmented = augmentor(image=img, bboxes=[coords], class_labels=['right_hand'])

# print(augmented)
# print(augmented['bboxes'][0][2:])
# print(augmented['bboxes'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [1920, 1080]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [1920, 1080]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])
plt.show()
"""


# function that runs throw all data and albuments it
def albument():
    for partition in ['train', 'test', 'val']:
        for image in os.listdir(os.path.join('data', partition, 'images')):
            img = cv2.imread(os.path.join('data', partition, 'images', image))

            coords = [0, 0, 0.00001, 0.00001]
            label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [1920, 1080, 1920, 1080]))

            try:
                # modifies each photo 60 times
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                                augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'),
                              'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)


# albument()

# function for loading images and their labels into tf dataset

# helper functions
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


# some lines are commented because idea is to build 1920x1080 resolution detector
def load_to_tf():
    limit_gpu()

    # for loading images
    train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    # train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
    train_images = train_images.map(lambda x: x / 255)
    test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    # test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
    test_images = test_images.map(lambda x: x / 255)
    val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    # val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
    val_images = val_images.map(lambda x: x / 255)

    # for testing
    # print(train_images.as_numpy_iterator().next())
    # print(len(train_images), len(test_images), len(val_images))

    # for loading labels
    train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    # for testing
    # print(train_labels.as_numpy_iterator().next())
    # print(len(train_labels), len(test_labels), len(val_labels))

    # for combining
    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(4200)
    train = train.batch(8)
    train = train.prefetch(4)
    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(1200)
    test = test.batch(8)
    test = test.prefetch(4)
    val = tf.data.Dataset.zip((val_images, val_labels))
    val = val.shuffle(1140)
    val = val.batch(8)
    val = val.prefetch(4)

    # for testing
    # print(train.as_numpy_iterator().next()[1])

    # to view
    data_samples = train.as_numpy_iterator()
    res = data_samples.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

        ax[idx].imshow(sample_image)


load_to_tf()
