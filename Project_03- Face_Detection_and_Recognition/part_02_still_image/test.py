import cv2
import dlib
import imutils
import os
import numpy as np
detector = dlib.get_frontal_face_detector()
detector_cnn = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

def convert_and_trim_bb(image,rect):
    ## Extracting the starting and ending (x,y) coordinates of the bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    ## Ensure the bounding box coordinates fall within the spatial dimensions of the image
    startX = max(0,startX)
    startY = max(0,startY)
    endX = max(0,endX)
    endY = max(0,endY)

    ## Compute the widht and height of the bounding box
    w = endX - startX
    h = endY - startY
    #return the bouding area
    return (startX, startY, w,h)

def detect_face(mode,image_path,window_name='default'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found: %s" % image_path)
    conv_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    text = None
    color = None
    if mode == 'cnn':
        rects = detector_cnn(conv_img,1)
        faces = [convert_and_trim_bb(conv_img,r.rect) for r in rects]
        text ="MMOD"
        color = (0,0,255)
    elif mode == 'hog':
        rects = detector(conv_img,1)
        faces = [convert_and_trim_bb(conv_img,r) for r in rects]
        text ="HOG"
        color = (255,0,0)
    
    for (x,y,w,h) in faces:
        img  = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.namedWindow(f"{window_name} Detected")
    cv2.imshow(f'{window_name}',img)
    cv2.imwrite(f'{window_name}_face_detection.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_face(mode='hog',image_path="./me2.jpg",window_name="HOG")