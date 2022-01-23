import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
print('Loading model')
model = tf.keras.models.load_model('models/09-0.11.hdf5')
print('Model loaded')


def decode_class(one_hot):
    cls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']
    return cls[np.argmax(one_hot)]


last_char = 'None'
i = 1001
letter = 'Z'
text = ''
char_time_start = time.time()
while (True):
    ret, frame = vid.read()
    batch = cv2.resize(frame, (200, 200))
    # cv2.imwrite(f'data/raw/asl-own/{letter}/{letter}_own_{i}.png', batch)
    batch = np.reshape(batch, (1, 200, 200, 3))
    batch = batch * (1 / 255.)
    pred = model.predict(batch, verbose=0)
    char = decode_class(pred)
    if char != last_char:
        last_char = char
        char_time_start = time.time()
    if (time.time() - char_time_start) > 1.0:
        text += char
        char_time_start = time.time()
    frame = cv2.putText(frame,
                        f'Char: {char}',
                        org=(1, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=1,
                        color=(0, 0, 255))
    frame = cv2.putText(frame,
                        f'Text:{text if len(text) < 10 else text[-10:]}',
                        org=(1, 55),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=1,
                        color=(0, 255, 0))
    cv2.imshow('frame', frame)
    # i += 1
    if cv2.waitKey(1) & 0xFF == ord('q') or i > 2000:
        break

vid.release()
cv2.destroyAllWindows()
