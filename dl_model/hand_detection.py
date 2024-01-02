from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

handtracer = load_model('handtracer.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    # frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (256, 144))

    yhat = handtracer.predict(np.expand_dims(resized / 255, 0))
    print(yhat)
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
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

        # Controls the text rendered
        cv2.putText(frame, 'right-hand', tuple(np.add(np.multiply(sample_coords[:2], [1920, 1080]).astype(int),
                                                      [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('HandTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
