import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
cam=cv2.VideoCapture(0);
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255,255,255)
conf=0

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (conf < 100):
             conf= "  {0}%".format(conf)
             
             if(id==1):
                id="Jackie Chan"
             elif(id==2):
                id="Timberlake"
             elif(id==3):
                id="obama"
             elif(id==4):
                 id="Benedict cumberbatch"
             else:
                id="tanımlanmayan cisim"
                conf= "  {0}%".format(conf)

        cv2.putText(img,str(id),(x,y+h),font,fontscale,fontcolor);
        cv2.putText(img,str(conf),(x+w-h,y),font,fontscale,fontcolor);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
