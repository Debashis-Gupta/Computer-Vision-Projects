import cv2
import dlib
import imutils


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

detector = dlib.get_frontal_face_detector()
detector_cnn = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
camera = cv2.VideoCapture(0)
while(cv2.waitKey(2)== -1):
    
    success,frame = camera.read()
    color  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rects = detector(color,1)
    faces = [convert_and_trim_bb(color,r) for r in rects]
    rects_cnn= detector_cnn(color,1)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, 'HOG', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    faces_cnn = [convert_and_trim_bb(color,r.rect) for r in rects_cnn]
    for (x,y,w,h) in faces_cnn:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame, 'MMOD_CNN', (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
    cv2.imshow("Face Detection",frame)    
