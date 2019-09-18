# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:37:37 2019

@author: SHEFALI MANGAL
"""

import cv2
import numpy as np
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/trainingData.yml")
id=0
#font=cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3

fontColor = (0,0,255)




faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,60),3)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (id==1):
            id=("Shefali")
        elif (id==2):
               id=("Bidusha")
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255),2);
        cv2.putText(img,str(id),(x,y+h), fontFace, fontScale,fontColor) 

    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
                           
