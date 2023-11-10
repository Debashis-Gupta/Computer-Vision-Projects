import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')

def create_alpha(image):
    if image.shape[2] == 3:
        with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)
        with_alpha[:, :, :3] = image
        with_alpha[:, :, 3] = 127
        return with_alpha

bunny_ears = cv2.imread('ears.jpeg', -1)
bunny_nose = cv2.imread('nose_bunny.jpg', -1)  # Assuming you have a separate image for the nose
bunny_glass = cv2.imread('glass.png', -1)
if bunny_ears.shape[2] < 4: 
    bunny_ears = create_alpha(bunny_ears)
if bunny_nose.shape[2] < 4:
    bunny_nose = create_alpha(bunny_nose)

if bunny_glass.shape[2] < 4:
    bunny_glass = create_alpha(bunny_glass)

camera = cv2.VideoCapture(0)
while cv2.waitKey(1) == -1:
    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

        for (x, y, w, h) in faces:
            # Overlay bunny ears
            ear_width = w
            ear_height = int(0.5 * h)
            ear_y = max(0, y - ear_height + int(0.05 * h))
            ear_end_y = min(frame.shape[0], ear_y + ear_height)
            actual_ear_height = ear_end_y - ear_y
            resized_ears = cv2.resize(bunny_ears, (ear_width, actual_ear_height))
            for c in range(0, 3):
                frame[ear_y: ear_end_y, x: x + ear_width, c] = frame[
                    ear_y: ear_end_y, x: x + ear_width, c
                ] * (1 - resized_ears[:, :, 3] / 255.0) + resized_ears[:, :, c] * (resized_ears[:, :, 3] / 255.0)

            # Overlay bunny nose
            nose_width = int(0.2 * w)
            nose_height = int(0.1 * h)
            nose_x = x + int(0.4 * w) - int(0.1 * nose_width)
            nose_y = max(0, y + int(0.50 * h) - int(0.1 * nose_height))
            nose_end_y = min(frame.shape[0], nose_y + nose_height)
            actual_nose_height = (nose_end_y - nose_y)
            resized_nose = cv2.resize(bunny_nose, (nose_width, actual_nose_height))
            for c in range(0, 3):
                frame[nose_y: nose_end_y, nose_x: nose_x + nose_width, c] = frame[
                    nose_y: nose_end_y, nose_x: nose_x + nose_width, c
                ] * (1 - resized_nose[:, :, 3] / 255.0) + resized_nose[:, :, c] * (resized_nose[:, :, 3] / 255.0)

            roi_gray = gray[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        cv2.imshow("Face Detection with Bunny Filter", frame)


# import cv2
# import numpy as np
# face_cascade = cv2.CascadeClassifier(
#     '../cascade/haarcascade_frontalface_default.xml'
# )
# eye_cascade = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')

# def create_alpha(image):
#     # Check if the image already has an alpha channel
#     if image.shape[2] == 3:
#         # Create a new image with an alpha channel
#         with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)

#         # Copy the RGB data from the original image
#         with_alpha[:, :, :3] = image

#         # Set the entire alpha channel to a specific value (e.g., 127 for half transparency)
#         with_alpha[:, :, 3] = 127
#         return with_alpha
#         # Save the image with alpha channel
#         # cv2.imwrite('path_to_save_with_alpha.png', with_alpha)

# # Load the filter images
# bunny_ears = cv2.imread('ears.jpeg', -1)
# bunny_nose = cv2.imread('ears.jpeg', -1)
# if bunny_ears.shape[2] < 4: 
#     bunny_ears = create_alpha(bunny_ears)
    

# if bunny_nose.shape[2] < 4:
#     bunny_nose = create_alpha(bunny_nose)
#     # raise ValueError("The bunny nose image does not have an alpha channel.")
# camera = cv2.VideoCapture(0)
# while cv2.waitKey(1) == -1:
#     success, frame = camera.read()
#     if success:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))

#         for (x, y, w, h) in faces:
#             # Overlay bunny ears
#             ear_width = w
#             ear_height = int(0.5 * h)
#             ear_y = y - ear_height + int(0.05 * h)
#             resized_ears = cv2.resize(bunny_ears, (ear_width, ear_height))
#             for c in range(0, 3):
#                 frame[ear_y: ear_y + ear_height, x: x + ear_width, c] = frame[
#                     ear_y: ear_y + ear_height, x: x + ear_width, c
#                 ] * (1 - resized_ears[:, :, 3] / 255.0) + resized_ears[:, :, c] * (resized_ears[:, :, 3] / 255.0)

#             # Overlay bunny nose
#             nose_width = int(0.2 * w)
#             nose_height = int(0.1 * h)
#             nose_x = x + int(0.4 * w) - int(0.1 * nose_width)
#             nose_y = y + int(0.45 * h) - int(0.1 * nose_height)
#             resized_nose = cv2.resize(bunny_nose, (nose_width, nose_height))
#             for c in range(0, 3):
#                 frame[nose_y: nose_y + nose_height, nose_x: nose_x + nose_width, c] = frame[
#                     nose_y: nose_y + nose_height, nose_x: nose_x + nose_width, c
#                 ] * (1 - resized_nose[:, :, 3] / 255.0) + resized_nose[:, :, c] * (resized_nose[:, :, 3] / 255.0)

#             roi_gray = gray[y: y + h, x: x + w]
#             eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40, 40))
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

#         cv2.imshow("Face Detection with Bunny Filter", frame)