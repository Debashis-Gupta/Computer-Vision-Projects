import cv2
import numpy as np
# Load the cascades
face_cascade = cv2.CascadeClassifier('../cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../cascade/haarcascade_eye.xml')



# Load the images with alpha channel
ears_image = cv2.imread('hair.png', -1)
nose_image = cv2.imread('fancy_nose.png', -1)  # Assuming you have a separate image for the nose
glasses_image = cv2.imread('dragon_eye.png', -1)
def create_alpha(image):
    if image.shape[2] == 3:
        with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)
        with_alpha[:, :, :3] = image
        with_alpha[:, :, 3] = 127
        return with_alpha
    
if ears_image.shape[2] < 4: 
    ears_image = create_alpha(ears_image)
if nose_image.shape[2] < 4:
    nose_image = create_alpha(nose_image)

if glasses_image.shape[2] < 4:
    glasses_image = create_alpha(glasses_image)

# Overlay the range  of the image
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


# Resizing the image with maintaining the aspect ratio of resizing the image
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame2 = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # Face detected, now detect eyes in the face ROI
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 3)
        
        # Draw bunny hair
        ears_resized = resize_with_aspect_ratio(ears_image, width=int(w*2))
        ew, eh, ec = ears_resized.shape
        frame = overlay_image_alpha(frame2,
                                    ears_resized[:, :, 0:3],
                                    (x-170, y - eh+330),
                                    ears_resized[:, :, 3] / 255.0)
        
        # Draw glasses on eyes
        if len(eyes) == 2:
            # Assuming the first two detections are the eyes
            for (ex, ey, ew, eh) in eyes[:2]:
                glasses_resized = resize_with_aspect_ratio(glasses_image, width=ew*2)
                gw, gh, gc = glasses_resized.shape
                frame = overlay_image_alpha(frame2,
                                            glasses_resized[:, :, 0:3],
                                            (x + ex-15, y + ey),
                                            glasses_resized[:, :, 3] / 255.0)

        # Draw bunny nose
        nose_resized = resize_with_aspect_ratio(nose_image, width=int(w/2))
        nw, nh, nc = nose_resized.shape
        nx = x + (w - nw) // 2
        ny = y + (h // 2)
        frame = overlay_image_alpha(frame2,
                                    nose_resized[:, :, 0:3],
                                    (nx, ny-90),
                                    nose_resized[:, :, 3] / 255.0)

    cv2.imshow('Bunny Filter', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
