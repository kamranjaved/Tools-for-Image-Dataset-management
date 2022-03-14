import numpy as np
import os
import glob
from skimage import io, color

imgA_list = './T.txt'
imgB_list = './I.txt'
imgC_list = './M.txt'
out_dir = './CmbS'
num_img = 5000

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
listA = open(imgA_list, 'r')
listB = open(imgB_list, 'r')
listC = open(imgC_list, 'r')

for i in range(num_img):
    if i%100 == 0:
        print(i)
    fileA = listA.readline()[:-1]
    filenameA = os.path.split(fileA)[-1]
    imgA = io.imread(fileA)
    print(filenameA)
    if imgA.shape[2] < 3:
        imgA = color.gray2rgb(imgA)
    fileB = listB.readline()[:-1]
    filenameB = os.path.split(fileB)[-1]
    print(filenameB)
    imgB = io.imread(fileB)
    print('B',imgB.shape)
    if imgB.shape[2] < 3:
        imgB = color.gray2rgb(imgB)

    fileC = listC.readline()[:-1]
    filenameC = os.path.split(fileC)[-1]
    imgC = io.imread(fileC)
    print('Cbefore',imgC.shape)
    imgC = np.expand_dims(imgC, axis=2)
    #if imgC.shape[2] < 3:
    #   imgC = color.gray2rgb(imgC)
    #imgC = np.atleast_3d(imgC)
    print('c after', imgC.shape)

    imgD = np.concatenate((imgC, imgC,  imgC),axis=2)
    print('A',imgA.shape)
    print('B',imgB.shape)
    print('C',imgC.shape)
    print('D',imgD.shape)
    

    out_img = np.concatenate((imgB, imgA,  imgD),axis=1)
    io.imsave(out_dir+'/'+filenameA, out_img)

    imgA = imgB = imgC = imgD = 0


