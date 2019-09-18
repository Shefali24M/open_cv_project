# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:53:07 2019

@author: SHEFALI MANGAL
"""

#import cv2
#import numpy as np
import cv2
import numpy as np

#create variable

#faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default');
#faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');


#create video capture object

#cam=cv2.VideoCapture(0); #video capture id ,any no. can be used
cam=cv2.VideoCapture(0);


#capture the frame one by one and detect the faces and show them into window so we apply loop

#while(True):
while(True):
    #ret,img=cam.read();
    ret,img=cam.read();

    #cam.read return status variable and captured image, but image()img gives coloured image
    #but for classifier to work we want grayscaled image, so we convert colured image into grayscaled image.
#    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #now we have grayscale image so we detect faces from that
    #now store the faces 
  #  faces=faceDetect.detectMultiScale(gray,1.3,5);
#    faces=faceDetect.detectMultiScale(gray,1.3,5);
    faces=faceDetect.detectMultiScale(gray,1.3,5);

    
    #store all the faces and provide coordinates of the faces.
    #as there are many faces in the faces variable so we draw rectangle around each faces, for that we use for loop.
    
    #for(x,y,w,h) in faces:
    for(x,y,w,h) in faces:
    
#        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

        #in this img must be coloured bcz we want to draw rect on colour image
        #now showing the image into the window.
    #    cv2.imshow("Face",img);
        cv2.imshow("Face",img);

        #before closing we need to give wait command otherwise open cv wont work.
        
        #if(cv2.waitKey(1)==ord('q')):
        if(cv2.waitKey(1)==ord('q')):
            break;
        
#cam.release() #to release the camera
#cv2.destroyAllWindows()
cam.release()
cv2.destroyAllWindows()   

