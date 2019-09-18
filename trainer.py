# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:41:52 2019

@author: SHEFALI MANGAL
"""

import os
import cv2
import numpy as np
import pil
from pil import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();

#recognizer=cv2.createLBPHFaceRecognizer();
#recognizer = cv2.face.createLBPHFaceRecognizer();
#recognizer = cv2.face.LBPHFaceRecognizer_create()


path='dataSet'
#to get the image we need corresponding ids, so we create method
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print (imagePaths)

#getImagesWithID(path)    
    
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        # this is pil image so we need to convert it into numpy array so that open cv can work with it
        faceNp=np.array(faceImg,'uint8')

        #now we want face ids
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        #now we have id and images now we can directly store it to into the faces and ids.
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp) #to show which images  that are captured.
        cv2.waitKey(70)
        
    return np.array(IDs), faces
IDs,faces=getImagesWithID(path)
#now we have faces ids , now we can train the recognizer
recognizer.train(faces,IDs)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()

