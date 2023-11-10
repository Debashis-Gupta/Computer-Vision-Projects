import cv2 
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(
    '../cascade/haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')
# output_folder = os.path.basename(os.getcwd())
# print(output_folder)



def create_data(label_name):
    output_folder = f"output/{label_name}/"
    if os.path.exists(output_folder):
        print("Image already exists")
        return
    else:
        os.makedirs(output_folder)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    
    camera = cv2.VideoCapture(0)
    count = 0
    while(cv2.waitKey(1) == -1):
        success,frame = camera.read()
        if success:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5,minSize=(120,120))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face_img =cv2.resize(gray[y:y+h,x:x+w],(200,200))
                face_filename = '%s/%d.pgm' % (output_folder,count)
                cv2.imwrite(face_filename,face_img)
                count +=1
                
            cv2.imshow('Capturing Faces ....' ,frame)

def read_images(path,image_size):
    print("Reading images........")
    names =[]
    training_images,training_labels =[],[]
    label =0

    for dirname,subdirnames,filenames in os.walk(path):
        
        for subdirname in subdirnames:
            # print(f"subdirname : {subdirname}")
            names.append(subdirname)
            subject_path = os.path.join(dirname,subdirname)
            # print(f"Subject path : {subject_path}")
            for filename in os.listdir(subject_path):
                # print(f"filename : {filename}")
                img = cv2.imread(os.path.join(subject_path,filename),cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img,image_size)
                training_images.append(img)
                training_labels.append(label)
            label +=1
    training_images = np.asarray(training_images,np.uint8)
    training_labels = np.asarray(training_labels,np.int32)
    return names,training_images,training_labels

def recognize_video(model,names,training_image_size,target_label,threshold):
    print("Starting recognition........")
    camera = cv2.VideoCapture(0)
    # actual_label_list = list()
    # predicted_label_list = list()
    TP=0
    FP=0
    TN=0
    FN=0
    count=0
    while(cv2.waitKey(1)==-1):
        success,frame = camera.read()
        if success:
            faces = face_cascade.detectMultiScale(frame,1.3,5)
            count+=1
            if(len(faces)<1):
                TN+=1
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                roi_gray = gray[x:x+w,y:y+h]
                if roi_gray.size == 0:
                    continue
                roi_gray = cv2.resize(roi_gray,training_image_size)
                predicted_label,confidence = model.predict(roi_gray)
                if predicted_label == target_label:
                    # predicted_label_list.append(predicted_label)
                    TP+=1
                else:
                    FP+=1
                    # predicted_label_list.append(-1)

                # actual_label_list.append(target_label)
                
                if confidence < threshold:
                    text = 'predicted_label = %s person=%s, confidence=%.2f' % (predicted_label,names[predicted_label],confidence)
                else:
                    text ="Unknown"
                    FN+=1
                
                cv2.putText(frame,text,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Recognition',frame)
    # print(f"Actual_label_list:{actual_label_list}\nPredicted_label_list:{predicted_label_list}")
    # return actual_label_list,predicted_label_list
    print("Total Frame: %s" % count)
    return TP,TN,FP,FN

def model_training(target_label,classifier):
    print("Model training....")
    path_to_training_image = 'output'
    training_image_size = (200,200)
    names,training_images,training_labels = read_images(path=path_to_training_image,image_size=training_image_size)
    target_label = names.index(target_label)
    if classifier=="Eigen":
        model = cv2.face.EigenFaceRecognizer_create()
        threshold =11555
    elif classifier=="Fisher":
        model = cv2.face.FisherFaceRecognizer_create()
        threshold = 1000
    else:
        model=cv2.face.LBPHFaceRecognizer_create()
        threshold=75
    model.train(training_images,training_labels)
    # for i in range(5):
    #     model.train(training_images,training_labels)
    #     print(f"Successfully trained for iteration {i+1}")
    print(names)
    print(len(training_images))
    print(len(training_labels))
    # model.save("trained_face.xml")
    return recognize_video(model,names,training_image_size,target_label,threshold)

if __name__ == "__main__":
    model_name=input("Enter 'e' for eigen , 'f' for fisher or 'l' for LBH classifier: ")

    if model_name.lower() == "e":
        classifier ="Eigen"
    elif model_name.lower() == "f":
        classifier ="Fisher"
    elif model_name.lower() == "l":
        classifier="LBH"
    else:
        exit("BAD ARGUMENT | TYPE e, f or l for selecting the model")
    
    scan = input("Enter y or yes if you want to add into dataset Or Type anything: ")
    if scan.lower() == 'y' or scan.lower() == 'yes':
        label = input("Enter the name of the person: ")
        create_data(label.lower())
    TP,TN,FP,FN= model_training(target_label='debashis',classifier=classifier)
    
    print(f"True Positive-{TP}\nFalse Negative-{FN}\nFalse Positive-{FP}\nTrue Negative-{FN}")

    Recall = TP / (TP+FN)
    Precision = TP / (TP+FP)
    Accouracy = (TP + TN) / (TP+TN+FP+FN)
    Specificity = TN/(FP+TN)  # TRUE_NEGATIVE_RATE 
    False_Positive_Rate = 1 - Specificity
    False_Negative_Rate = FN/(TP+FN)
    F1_score = (2*Precision*Recall)/(Precision+Recall)
    print(f"Recall = {Recall} and Precision = {Precision}")
    print(f"Accouracy = {Accouracy}")
    print(f"Specificity = {Specificity}")
    print(f"False_Positive_Rate = {False_Positive_Rate}")
    print(f"False_Negative_Rate = {False_Negative_Rate}")
    print(f"F1_score = {F1_score}")
    
