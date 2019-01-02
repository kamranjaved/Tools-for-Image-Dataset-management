import numpy as np
import os
import dlib
import glob
from skimage import io, transform, data, color

predictor_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = './test'
out_path = './testa'
missing = out_path+'_fail_to_landmark.txt'
#dst = np.asarray([[48,63],[79,63]])
#dst = np.asarray([[96,126],[158,126]])
dst = np.asarray([[170,240],[286,240]])
height = width = 472

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
txtfile = open(missing,'w')
src = np.zeros((2,2))
i=0

for fname in os.listdir(faces_folder_path):
    i = i+1
    if i % 100 == 0:
        print(i)
    f = faces_folder_path + '/' + fname
    img = io.imread(f)
    img = color.gray2rgb(img)

    #win.clear_overlay()
    #win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    if len(dets) != 1:
        txtfile.write("%s\n" % fname)
        continue
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # source pts
        src[0][0] = (shape.part(37).x+shape.part(38).x+shape.part(40).x+shape.part(41).x)/4.0
        src[0][1] = (shape.part(37).y+shape.part(38).y+shape.part(40).y+shape.part(41).y)/4.0
        src[1][0] = (shape.part(43).x+shape.part(44).x+shape.part(46).x+shape.part(47).x)/4.0
        src[1][1] = (shape.part(43).y+shape.part(44).y+shape.part(46).y+shape.part(47).y)/4.0
        #src[2][0] = (shape.part(61).x+shape.part(64).x)/2.0
        #src[2][1] = (shape.part(61).y+shape.part(64).y)/2.0
        #src = src.astype('int32')
        
        tform=transform.estimate_transform('similarity', src, dst)
        warped = transform.warp(img, tform.inverse, output_shape=(height, width, 3))
        out_img=warped
        #out_img = warped[:height,:width]#*255
        #out_img = out_img.astype('uint8')

        io.imsave(out_path+'/'+fname,out_img)
        
        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)

    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()

