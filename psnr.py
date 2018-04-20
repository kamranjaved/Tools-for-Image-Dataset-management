import math
import glob
import os
import cv2
import numpy as np

ratio = 0.90
image_size = 128

x = []
y = []
psnr= []
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

    print('No of total images = %d'%np.size(x,0))
    for i in range (np.size(x,0)):
          mse = np.mean( (x[i] - y[i]) ** 2 )
          if mse == 0:
             return 100
          PIXEL_MAX = 255.0
          
          psnr.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
          #print('PSNR of image_%(i)d = %(psnr)f'% {'i':i+1, 'psnr':psnr[i]})
          i+=1
    avgpsnr = np.average(psnr)
    print('Average PSNR of %(i)d images = %(avgpsnr)f'% {'i':i, 'avgpsnr':avgpsnr})    
    return avgpsnr

if __name__ == '__main__':
    PSNR()
    
