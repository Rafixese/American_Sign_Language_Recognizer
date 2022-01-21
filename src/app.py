import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
i = 1
letter = 'Z'
while (True):
    ret, frame = vid.read()
    batch = cv2.resize(frame, (200, 200))
    # cv2.imwrite(f'data/raw/asl-own/{letter}/{letter}_own_{i}.png', batch)
    batch = np.reshape(batch, (1, 200, 200, 3))
    batch = batch * (1 / 255.)
    pred = model.predict(batch, verbose=0)
    frame = cv2.putText(frame,
                        f'Char: {decode_class(pred)}',
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255))
    cv2.imshow('frame', frame)
    # i += 1
    if cv2.waitKey(1) & 0xFF == ord('q') or i > 1000:
        break

vid.release()
cv2.destroyAllWindows()
