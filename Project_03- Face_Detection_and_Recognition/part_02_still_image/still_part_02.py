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

def check_folder_exists(folder_name):
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # The name of the directory to be checked or created
    output_directory_name = folder_name

    # Full path to the output directory
    output_directory_path = os.path.join(script_directory, output_directory_name)

    # Check if the directory exists
    if not os.path.exists(output_directory_path):
        # If it does not exist, create it
        os.makedirs(output_directory_path)
        print(f"Directory '{output_directory_name}' created at: {output_directory_path}")
    # else:
    #     print(f"Directory '{output_directory_name}' already exists at: {output_directory_path}")

def image_rotation(image,angle):
    height,width = image.shape[:2]
    center = (width/2,height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle,1.0)
    rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
    return rotated_image
# def invariant_check_face(img, isWood, method, window_name):
    
#     # Convert the image to grayscale for the detector
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # rects = detector(gray, 1)
#     # faces = [convert_and_trim_bb(gray, r) for r in rects]
#     rects= detector_cnn(gray,1)
#     faces = [convert_and_trim_bb(gray,r.rect) for r in rects]

    
#     # Draw the rectangle on the colored image 'img' not 'gray'
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # cv2.putText(img, 'HOG', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     # Display and save the colored image 'img' with rectangles
#     # cv2.namedWindow(f"{window_name} Detected?")
#     # cv2.imshow(f'{window_name}', img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     check_folder_exists(method) 
#     if method == 'rotation': 
#         if isWood:
#             cv2.imwrite(f'{method}/{window_name}_woodcutter_degree_face_detection.jpg', img)
#         else:
#             cv2.imwrite(f'{method}/{window_name}_me_degree_face_detection.jpg', img)
#     elif method == "scale":
#         if isWood:
#             cv2.imwrite(f'{method}/{window_name}_woodcutter_scale_face_detection.jpg', img)
#         else:
#             cv2.imwrite(f'{method}/{window_name}_me_scale_scale_detection.jpg', img)
#     elif method == 'translation':
#         if isWood:
#             cv2.imwrite(f'{method}/{window_name}_woodcutter_translation_face_detection.jpg', img)
#         else:
#             cv2.imwrite(f'{method}/{window_name}_me_translation_detection.jpg', img)
#     else:
#         print("Wrong Method Selection. Please input either rotation,scale or translation")
#         exit("Bad Arguments for method")

def invariant_check_face(img,cnn,method,window_name,):

    # color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    color = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    rects = detector(color,1)
    faces = [convert_and_trim_bb(color,r) for r in rects]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if cnn:
            cv2.putText(img, 'MMOD', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            cv2.putText(img, 'HOG', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    # cv2.namedWindow(f"{window_name} Detected?")
    # cv2.imshow(f'{window_name}',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    if method == 'rotation': 
        check_folder_exists(method)    
        if cnn:
            
            cv2.imwrite(f'{method}/{window_name}_me_degree_face_detection_cnn.jpg',img)
        else:
            cv2.imwrite(f'{method}/{window_name}_me_degree_face_detection.jpg',img)
    elif method =="scale":
        check_folder_exists(method)
        if cnn:
            cv2.imwrite(f'{method}/{window_name}_me_scale_face_detection_cnn.jpg',img)
        else:
            cv2.imwrite(f'{method}/{window_name}_me_scale_scale_detection.jpg',img)
    elif method == 'translation':
        check_folder_exists(method)
        if cnn:
            cv2.imwrite(f'{method}/{window_name}_me_translation_face_detection_cnn.jpg',img)
        else:
            cv2.imwrite(f'{method}/{window_name}_me_translation_detection.jpg',img)
         
    else:
        print("Wrong Method Selection. Please input either rotation,scale or translation")
        exit("Bad Arguments for method")



def rotate_invariant_face_detection(image_path, isWood,rotation_angles = [0, 45, 90, 135, 180]):
    img =cv2.imread(image_path)

    for rot in rotation_angles:
        rot_image = image_rotation(img,rot)
        invariant_check_face(rot_image,isWood,method='rotation',window_name=rot)


def scale_invariant_face_detection(image_path,isWood,scale_factors = [1.0, 1.5, 2.0,2.5]):
    img = cv2.imread(image_path)
    for  scale in scale_factors:
        scaled_image = cv2.resize(img, None, fx=scale, fy=scale)
        invariant_check_face(scaled_image,isWood,method='scale',window_name=scale)


def translation_invariant_face_detection(image_path,isWood,pixels_shift =[25,50,75,100,125,150,175,200]):
    img = cv2.imread(image_path)
    for px_shift in pixels_shift:
        M = np.float32([[1, 0, px_shift], [0, 1, px_shift*2]])  
        shifted_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))  ## shift the image 15 pixels to the right and 30 pixels down and continuous .......
        invariant_check_face(shifted_image,isWood,method='translation',window_name=px_shift)

if __name__ == "__main__":
    # detect_face('woodcutters.jpg','woodcutters')
    # detect_face('me2.jpg','me')
    # rotate_invariant_face_detection('me2.jpg',isWood=True)
    # rotate_invariant_face_detection('me2.jpg',isWood=False)
    # # rotate_invariant_face_detection('me2.jpg',isWood=True)
    # scale_invariant_face_detection('me2.jpg',isWood=True)
    # scale_invariant_face_detection('me2.jpg',isWood=False)
    # # scale_invariant_face_detection('me2.jpg',isWood=True)
    # translation_invariant_face_detection('me2.jpg',isWood=True)
    translation_invariant_face_detection('me2.jpg',isWood=False)
    # translation_invariant_face_detection('me2.jpg',isWood=True)




"""
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

"""