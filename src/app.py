import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
print('Loading model')
model = tf.keras.models.load_model('models/07-163.48.hdf5')
print('Model loaded')


def decode_class(one_hot):
    cls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    return cls[np.argmax(one_hot)]


last_char = 'None'
while (True):
    ret, frame = vid.read()
    batch = frame * (1 / 255.)
    batch = cv2.resize(batch, (200, 200))
    batch = np.reshape(batch, (1, 200, 200, 3))
    pred = model.predict(batch)
    print(decode_class(pred))
    frame = cv2.putText(frame,
                        f'Char: {decode_class(pred)}',
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 0))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
