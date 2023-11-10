import cv2
import numpy

import utils

##### CODES NEEDED FOR FILTERING #####
# THE FOLLOWING CODE IS FOR APPLYING HISTOGRAM EQUALIZATION TO THE FRAME OF THE VIDEO
def filter_hist(src,dst):
        # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src= cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2YCR_CB)
    src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])
    src = cv2.cvtColor(src, cv2.COLOR_YCR_CB2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    cv2.merge([src], dst)
# THE FOLLOWING CODE IS USED FOR DETECTING EDGES IN THE FRAME OF THE VIDEO
def edgeFilter(src,dst):
    src = cv2.Canny(src, 50, 200)
    cv2.merge([src, src, src], dst)

# THE FOLLOWING CODE IS USED FOR UNSHARPENING THE FRAME OF THE VIDEO
def unsharp_filter(src,dst):
    src2 = cv2.GaussianBlur(src, (10, 10), 0)
    src = cv2.addWeighted(src,1.5,src2,-0.5,0)
    cv2.merge([src], dst)
# THE FOLLOWING CODE IS USED FOR SMOOTHING (BLURING) THE FRAME OF THE VIDEO
def smooth_filter(src,dst):
    src = cv2.blur(src, (20, 20))
    cv2.merge([src], dst)
# THE FOLLOWING CODE IS USED FOR ADAPTIVE FILTERING (CLAHE) THE FRAME OF THE VIDEO
def adaptive_filter(src,dst):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2YCR_CB)
    src[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(src[:, :, 0])
    src = cv2.cvtColor(src, cv2.COLOR_YCR_CB2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
    cv2.merge([src], dst)


##### END CODES NEEDED FOR FILTERING #####
