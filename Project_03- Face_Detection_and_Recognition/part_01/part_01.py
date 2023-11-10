import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(
    '../cascade/haarcascade_frontalface_default.xml'
)


def image_rotation(image,angle):
    height,width = image.shape[:2]
    center = (width/2,height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle,1.0)
    rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
    return rotated_image

def detect_face(image_path,window_name='default'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.08,5)
    for (x,y,w,h) in faces:
        img  = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.namedWindow(f"{window_name} Detected")
    cv2.imshow(f'{window_name}',img)
    cv2.imwrite(f'{window_name}_face_detection.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def invariant_check_face(img,isWood,method,window_name):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.08,5)
    for (x,y,w,h) in faces:
        img  = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.namedWindow(f"{window_name} Detected?")
    cv2.imshow(f'{window_name}',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    if method == 'rotation':     
        if isWood:
            cv2.imwrite(f'output_rot/{window_name}_woodcutter_degree_face_detection.jpg',img)
        else:
            cv2.imwrite(f'output_rot/{window_name}_me_degree_face_detection.jpg',img)
    elif method =="scale":
        if isWood:
            cv2.imwrite(f'output_scale/{window_name}_woodcutter_scale_face_detection.jpg',img)
        else:
            cv2.imwrite(f'output_scale/{window_name}_me_scale_scale_detection.jpg',img)
    elif method == 'translation':
        if isWood:
            cv2.imwrite(f'output_trans/{window_name}_woodcutter_translation_face_detection.jpg',img)
        else:
            cv2.imwrite(f'output_trans/{window_name}_me_translation_detection.jpg',img)
         
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
    # rotate_invariant_face_detection('woodcutters.jpg',isWood=True)
    # rotate_invariant_face_detection('me2.jpg',isWood=False)
    # scale_invariant_face_detection('woodcutters.jpg',isWood=True)
    # scale_invariant_face_detection('me2.jpg',isWood=False)
    translation_invariant_face_detection('woodcutters.jpg',isWood=True)
    translation_invariant_face_detection('me2.jpg',isWood=False)