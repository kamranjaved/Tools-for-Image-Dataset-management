import math
import glob
import os
from MSSSIM import * 
import cv2
import numpy as np
import tensorflow as tf

ratio = 0.90
image_size = 128

x = []
y = []
psnrval = []
ssimval = []
i=0
def PSNR():
    i=0
    #def npy():
    pathA = glob.glob('./A/*')
    pathB = glob.glob('./B/*')
    for path in pathA:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x.append(img)

    for path in pathB:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y.append(img)
    
    #ssim2 = tf.Session().run(psnr2(tf.convert_to_tensor(x[0]), tf.convert_to_tensor(y[0]), 255))
    
    #print(' PSNR by TF = ', ssim2 )
    txtfile = open('PSNR_and_SSIM.txt', 'w')
    print('No of total images = %d'%np.size(x,0))
    for i in range (np.size(x,0)):
          #mse = np.mean( (x[i] - y[i]) ** 2 )
          #if mse == 0:
          #   return 100
          #PIXEL_MAX = 255.0
          #p = 
          #s = 
          psnrval.append(tf.Session().run(psnr2(tf.convert_to_tensor(x[i]), tf.convert_to_tensor(y[i]), 255)))
          ssimval.append(tf.Session().run(ssim_multiscale(tf.convert_to_tensor(x[i]), tf.convert_to_tensor(y[i]), 255)))
          print('PSNR of image_%(i)d = %(psnrval)f'% {'i':i+1, 'psnrval':psnrval[i]}, file=txtfile)
          print('MS-SSIM of image_%(i)d = %(ssimval)f'% {'i':i+1, 'ssimval':ssimval[i]}, file=txtfile)
          i+=1

    avgpsnr = np.average(psnrval)
    print('Average PSNR of %(i)d images = %(avgpsnr)f'% {'i':i, 'avgpsnr':avgpsnr} , file=txtfile)    
    avgssim = np.average(ssimval)
    print('Average MSSSIM of %(i)d images = %(avgssim)f'% {'i':i, 'avgssim':avgssim} ,  file=txtfile)    
    
    #ssim1 = ssim_multiscale(tf.convert_to_tensor(x[1]), tf.convert_to_tensor(x[1]), 255)
    #ssim2 = tf.Session().run(ssim_multiscale(tf.convert_to_tensor(x[1]), tf.convert_to_tensor(y[1]), 255))
    #ssim2 = tf.Session().run(psnr2(tf.convert_to_tensor(x[0]), tf.convert_to_tensor(y[0]), 255))
    
    #print(' MS-SSIM = ', ssim2 )
    #print(' PSNR by me= ', psnr[0] )
    txtfile.close()
    return avgpsnr, avgssim

if __name__ == '__main__':
    PSNR()
    
