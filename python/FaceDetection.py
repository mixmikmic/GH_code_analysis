import cv2
import numpy as np
import pyttsx as voice
from PIL import Image
import os
import time

faceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
engine = voice.init()

id = raw_input("Enter your Id!!\n")
sampleNumber=0
engine.say("Please, Look at the Camera, your Facial Data Set are getting ready!!")
engine.runAndWait()

while True:
    cv2.waitKey(100)
    ret,img = cam.read()
    imggrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetection.detectMultiScale(imggrey,1.3,5)
    for (x,y,w,h) in faces:
        sampleNumber += 1
        cv2.imwrite("FaceDataSets/user."+str(id)+"."+str(sampleNumber)+".jpg",imggrey[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.waitKey(100)
    cv2.imshow("HomeScreen",img)
    k = cv2.waitKey(10)
    if (sampleNumber > 20):
        break
engine.say("Thank you for your corporation, your Facial Data Sets are Ready!")
engine.runAndWait()
cam.release()
cv2.destroyAllWindows()

recog = cv2.face.createLBPHFaceRecognizer()
path = "FaceDatasets"

def getImagesById(path):
    imgPath = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for path in imgPath:
        faceimg = Image.open(path).convert('L')
        faceNp = np.array(faceimg,'uint8')
        Id=int(os.path.split(path)[-1].split(".")[1])
        faces.append(faceNp)
        Ids.append(Id)
        cv2.imshow("Traning",faceNp)
        cv2.waitKey(100)
    return np.array(Ids),faces

Ids,faces = getImagesById(path)
recog.train(faces,np.array(Ids))
engine.say("Training  Completed  at  "+time.asctime( time.localtime(time.time()) )+", System is ready to predict the faces")
engine.runAndWait()
recog.save("Training/FaceDetectionTraining.yml")
cv2.destroyAllWindows()
        

id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
recog.load('Training/FaceDetectionTraining.yml')
while True:
    ret,img = cam.read()
    imggrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetection.detectMultiScale(imggrey,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        id = recog.predict(imggrey[y:y+h,x:x+w])
        if id == 1:
            id = "Harshal"
        elif id == 2:
            id = "Devashish"
        elif id == 3:
            id = "Abhishek"
        elif id == 4:
            id = "Mansi"
        else:
            id = "Unknown"
        cv2.putText(img,str(id),(int((x+w/2)),y+h),font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("Faces",img)
    
    key = cv2.waitKey(10)
    if key == 27:
        break
        
engine.say("Hello "+str(id)+", Nice to see you!!")
engine.runAndWait()        
cam.release()
cv2.destroyAllWindows()





