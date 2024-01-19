"""
This file is for collecting training data
Basically collection pictures
"""

import os
import time
import uuid
import cv2

# Path where pictures are saved
IMAGES_PATH = os.path.join('data', 'images')

# How many pictures are taken
number_images = 100

# Let's take some pictures
cap = cv2.VideoCapture(0)
for img_num in range(number_images):
    print('Collecting image {}'.format(img_num))
    ret, frame = cap.read()
    img_name = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(img_name, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
