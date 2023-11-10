# THis is working 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log10
# THE FOLLOWING FUNCTION USED TO COUNT THE INTENSITY RANGE OF THE IMAGE (JUST TO SHOW A DIFFERENT APPROACH THAN NP.HISTOGRAM)
def calc_vect_hist(img):
    out = np.bincount(img.ravel(),minlength=256)
    return out
# THE FOLLOWING FUNCTION IS USED TO APPLY HISTOGRAM EQUALIZATION TO THE IMAGE
def histEq(transformed_img):
    if len(transformed_img.shape) > 2 and transformed_img.shape[2]==3: #THIS IS FOR COLOR IMAGE
        imgY = transformed_img[:, :, 0]
        hist = calc_vect_hist(imgY.flatten())
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) *255 / (cdf.max() - cdf.min())
        equalizedY = cdf_normalized[imgY]
        transformed_img[:, :, 0] = equalizedY
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_YCR_CB2RGB).astype(np.uint8)
        return transformed_img
    else:
        hist,bins = np.histogram(transformed_img.flatten(),256,[0,256]) #GRAYSCALE IMAGE
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) *255 / (cdf.max() - cdf.min())
        transformed_img = cdf_normalized[transformed_img]
        return transformed_img.astype(np.uint8)

if __name__ == '__main__':
    
    img = cv2.imread('Data/Dog.jpeg')
    print(f"Original image shape: {img.shape}")
    if len(img.shape) > 2 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        tick = cv2.getTickCount()
        transformed_img = histEq(trans_img)
        toc = cv2.getTickCount()
        mine_diff = (toc - tick) / cv2.getTickFrequency()
        converted_img = cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
        imgY = converted_img[:, :, 0]
        tick = cv2.getTickCount()
        equalized_cv2_y = cv2.equalizeHist(imgY)
        toc = cv2.getTickCount()
        cv2_diff = (toc - tick) / cv2.getTickFrequency()
        converted_img[:, :, 0] = equalized_cv2_y
        equalized_cv_img = cv2.cvtColor(converted_img,cv2.COLOR_YCR_CB2RGB).astype(np.uint8)
        print("#"*10+"Proceesing Ploting Image"+ "#"*10)
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title('Original')
        plt.subplot(1,3,2)
        plt.imshow(transformed_img)
        plt.title('Equalized Mine')
        plt.subplot(1,3,3)
        plt.imshow(equalized_cv_img)
        plt.title('CV2 Equalized')
        plt.tight_layout()
        plt.savefig("images/Image_HistogramEqualization_color.jpeg",dpi=300)
        plt.show()
        print("#"*10+"Proceesing Ploting Histogram"+ "#"*10)
        plt.subplot(3,3,1)
        plt.hist(img[:,:,0].flatten(),256,[0,256],color='red')
        plt.title("Original Red Histogram")
        plt.subplot(3,3,2)
        plt.hist(transformed_img[:,:,0].flatten(),256,[0,256],color='red')
        plt.title("Mine Red Histogram")
        plt.subplot(3,3,3)
        plt.hist(equalized_cv_img[:,:,0].flatten(),256,[0,256],color='red')
        plt.title("CV2 Red Histogram")
        plt.subplot(3,3,4)
        plt.hist(img[:,:,1].flatten(),256,[0,256],color='green')
        plt.title("Original Green Histogram")
        plt.subplot(3,3,5)
        plt.hist(transformed_img[:,:,1].flatten(),256,[0,256],color='green')
        plt.title("Mine Green Histogram")
        plt.subplot(3,3,6)
        plt.hist(equalized_cv_img[:,:,1].flatten(),256,[0,256],color='green')
        plt.title("CV2 Green Histogram")
        plt.subplot(3,3,7)
        plt.hist(img[:,:,2].flatten(),256,[0,256],color='blue')
        plt.title("Original Blue Histogram")
        plt.subplot(3,3,8)
        plt.hist(transformed_img[:,:,2].flatten(),256,[0,256],color='blue')
        plt.title("Mine Blue Histogram")
        plt.subplot(3,3,9)
        plt.hist(equalized_cv_img[:,:,2].flatten(),256,[0,256],color='blue')
        plt.title("CV2 Blue Histogram")
        plt.tight_layout()
        plt.savefig("images/HistogramEqualization_color.jpeg",dpi=300)
        plt.show()
    else:
        tick = cv2.getTickCount()
        transformed_img= histEq(img)
        toc = cv2.getTickCount()
        mine_diff = (toc - tick) / cv2.getTickFrequency()
        tick = cv2.getTickCount()
        equalized_cv2_img = cv2.equalizeHist(img).astype(np.uint8)
        toc = cv2.getTickCount()
        cv2_diff = (toc - tick) / cv2.getTickFrequency()
        print("#"*10+"Proceesing Ploting Image"+ "#"*10)
        plt.subplot(1,3,1)
        plt.imshow(img,cmap='gray')
        plt.title('Original')
        plt.subplot(1,3,2)
        plt.imshow(transformed_img,cmap='gray')
        plt.title('Equalized Mine')
        plt.subplot(1,3,3)
        plt.imshow(equalized_cv2_img,cmap='gray')
        plt.title('CV2 Equalized')
        plt.tight_layout()
        plt.savefig("images/Image_HistogramEqualization_gray.jpeg",dpi=300,transparent=True)
        plt.show()
        print("#"*10+"Proceesing Ploting Histogram"+ "#"*10)    
        plt.subplot(3,1,1)
        plt.hist(img.flatten(),256,[0,256],color='black')
        plt.title("Original Image Histogram")
        plt.subplot(3,1,2)
        plt.hist(transformed_img.flatten(),256,[0,256],color='black')
        plt.title("Mine Image Histogram")
        plt.subplot(3,1,3)
        plt.hist(equalized_cv2_img.flatten(),256,[0,256],color='black')
        plt.title("CV2 Image Histogram")
        plt.savefig("images/HistogramEqualization_gray.jpeg",dpi=300,transparent=True)
        plt.tight_layout()
        plt.show()

    print("#"*10+"Proceesing Time"+ "#"*10)
    print(f"Time - Mine: {mine_diff}s")
    print(f"Time - CV2: {cv2_diff}s")
    print(f"Time - Mine: {abs(log10(mine_diff))}s")
    print(f"Time - CV2: {abs(log10(cv2_diff))}s")
    print(f"Log Difference: {abs(log10(mine_diff)-log10(cv2_diff))}s")
    
        
        
