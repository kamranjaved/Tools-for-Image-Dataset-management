import os
import glob
import cv2
import numpy as np

IMAGE_SIZE = 96


def load():
    dir_img = './DB256o/train/*'
    ratio = 0.90
    x = []
    paths = glob.glob(dir_img)
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x.append(img)

    x = np.array(x, dtype=np.uint8)
    np.save('x_train.npy', x)
    print('Shape of array', x.shape)
    return x


if __name__ == '__main__':
    load()
