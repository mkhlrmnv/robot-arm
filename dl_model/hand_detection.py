from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

handtracer = load_model('hadntracer.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    # frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (256, 144))

    yhat = handtracer.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    max_co = (1920 / 2, 1080 / 2)
    min_co = max_co

    if yhat[0] > 0.7:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [1920, 1080]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [1920, 1080]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [1920, 1080]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [1920, 1080]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        min_co = tuple(np.multiply(sample_coords[:2], [1920, 1080]).astype(int))
        max_co = tuple(np.multiply(sample_coords[2:], [1920, 1080]).astype(int))

        # Controls the text rendered
        cv2.putText(frame, 'hand', tuple(np.add(np.multiply(sample_coords[:2], [1920, 1080]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    centerX = int((max_co[0] + min_co[0]) / 2)
    centerY = int((max_co[1] + min_co[1]) / 2)
    print(f"Center X: {centerX} and center Y: {centerY}")

    cv2.circle(frame, (centerX, centerY), 10, (0, 0, 255), 2)

    cv2.imshow('HandTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
