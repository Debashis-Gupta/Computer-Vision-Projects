import cv2 
import numpy as np
face_cascade = cv2.CascadeClassifier(
    '../cascade/haarcascade_frontalface_default.xml'
)

recog_descriptor = list()

def learn_image(image_list):
    for img_path in image_list:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.08,5)
        for (x,y,w,h) in faces:
            face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
        recog_descriptor.append(face_img)


def run_recognition(recog_descriptor):
    camera = cv2.VideoCapture(0)
    while(cv2.waitKey(2)==-1):
        success,frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.08,5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
            difference = np.abs(face_img-recog_descriptor[0])
            result = np.all(difference < 0.25)
            
            if result:
                cv2.putText(frame, 'ME! Hello', (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("Face Detection",frame)


# img = cv2.imread("me2.jpg")
learn_image(["me.jpg"])
print(recog_descriptor)
run_recognition(recog_descriptor)
