# This is working for the convolution of an image with a kernel.
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2

def cross_check(rolling_view):
    first_view = rolling_view[0]
    print(f"Shape of first_view : {first_view.shape}")
    # print(f"First Rolling view : {first_view}")
    first_view_hist = histequalize(first_view[:,:,0])
    whole_thing = histequalize(rolling_view)
    print(f"Shape of whole_thing : {whole_thing.shape}")
    print(f"Shape of whole_thing[0] : {whole_thing[0].shape}")
    # print(f"first of whole_thing : {whole_thing[0]}")
    # print(first_view==whole_thing[0])
    plt.subplot(1,2,1)
    plt.hist(first_view_hist.flatten(),256,[0,256])
    plt.subplot(1,2,2)
    plt.hist(whole_thing[0].flatten(),256,[0,256])
    plt.show()

# # THIS FUNCTION IS FOR HISTOGRAM EQUALIZATION FOR EACH WINDOW SECTION OF KERNEL SIZE
def histequalize(transformed_img):
    hist,bins = np.histogram(transformed_img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    # print(f"Shape of transformed_img : {transformed_img.shape}")
    cdf_normalized = (cdf - cdf.min()) *255 / (cdf.max() - cdf.min())
    # print(f"Shape of cdf_normalized : {cdf_normalized.shape}")
    transformed_img = cdf_normalized[transformed_img]
    # print(f"Shape of transformed_img after cdf normalized : {transformed_img.shape}")
    return transformed_img
   
    
def adaptiveHistEq(input_image, kernel):
    # print(f"Shape of input_image : {input_image.shape}")
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        kernel_height, kernel_width =  kernel.shape
        
        # Calculate padding size for each dimension
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
       
        transY = cv2.cvtColor(input_image,cv2.COLOR_RGB2YCR_CB)
        imgY = transY[:,:,0]
        # print(f"Shape of imgY : {imgY.shape}")
        # for i in range(num_channels):
        # Apply padding to the input image
        padded_input = np.pad(imgY, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
        # print(f"Shape of padded_input : {padded_input.shape}")
            # Use sliding_window_view to create a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
        # print(f"Shape of rolling_view : {rolling_view.shape}")
        # print(f"Rolling_View : {rolling_view}")
        # cross_check(rolling_view)
        # exit()
        output = histequalize(rolling_view)
        # print(f"Shape of output : {output.shape}")
        new_img = np.empty((input_image.shape[0], input_image.shape[1]))
        new_img = output[:,:,1,1]
       
        transY[:,:,0] = new_img
        transY = cv2.cvtColor(transY,cv2.COLOR_YCR_CB2RGB).astype(np.uint8)
        return transY
            # Perform element-wise multiplication and sum to get the adaptiveHistEq result

    else:
        # print("GrayScale image")
        kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size for each dimension
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        
    
        # Apply padding to the input image
        padded_input = np.pad(input_image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
        # print(f"Shape of padded_input : {padded_input.shape}")
            # Use sliding_window_view to create a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
        # print(f"Shape of rolling_view : {rolling_view.shape}")
            # Perform element-wise multiplication and sum to get the adaptiveHistEq result
        # first_view = rolling_view[:,:,0]
        output=histequalize(rolling_view)
        # print(f"Shape of output : {output.shape}")
        new_img = np.empty((input_image.shape[0], input_image.shape[1]))
        new_img = output[:,:,1,1]
        
        # print(f"Shape of output : {output.shape}")

        return new_img.astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread("Data/Dog.jpeg")
    

    # Define the 3x3 kernel filter RANDOM INITIALIZATION - CAN BE DEFINED ANYTHING - WHAT MATTERS - ONLY THE SHAPE NOT THE VALUE
    kernel3x3 = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    

    # # Build-in CLAHE approach in CV
    if len(img.shape) == 3 and img.shape[2] == 3 :
        tick=cv2.getTickCount()
        convd_image = adaptiveHistEq(img, kernel3x3)
        toc = cv2.getTickCount()
        my_diff = (toc - tick) / cv2.getTickFrequency()
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y,cr,cb = cv2.split(ycrcb)
        tick = cv2.getTickCount()
        clahe = cv2.createCLAHE(clipLimit=101.0, tileGridSize=(3,3))
        y_clahe = clahe.apply(y)
        toc = cv2.getTickCount()
        cv2_diff = (toc - tick) / cv2.getTickFrequency()

        #merging the channels back to YCrCb
        ycrcb_clahe = cv2.merge([y_clahe,cr,cb])
        #back to RGB to BGR 
        img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
       
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title("Original image")
        plt.subplot(1,3,2)
        plt.imshow(convd_image)
        plt.title("Mine Adaptive Image")
        plt.subplot(1,3,3)
        plt.imshow(img_clahe)
        plt.title("CLAHE image")
        plt.tight_layout()
        plt.savefig("images/Adaptive_image_Color.jpeg",dpi=300)
        plt.show()
        plt.subplot(1,3,1)
        plt.hist(img.flatten(),256,[0,256])
        plt.title("Original image")
        plt.subplot(1,3,2)
        plt.hist(convd_image.flatten(),256,[0,256])
        plt.title("Mine Adaptive Image")
        plt.subplot(1,3,3)
        plt.hist(img_clahe.flatten(),256,[0,256])
        plt.title("CLAHE image")
        plt.tight_layout()
        # plt.savefig("images/Color_Adaptive_image_hist_equalization.jpeg",dpi=300)
        plt.show()
    else:
        tick=cv2.getTickCount()
        convd_image = adaptiveHistEq(img, kernel3x3)
        toc = cv2.getTickCount()
        my_diff = (toc - tick) / cv2.getTickFrequency()
        tick = cv2.getTickCount()
        clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(3,3))
        img_clahe = clahe.apply(img)
        toc = cv2.getTickCount()
        cv2_diff = (toc - tick) / cv2.getTickFrequency()
    # img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR)
        plt.subplot(1,3,1)
        plt.imshow(img,cmap='gray')
        plt.title("Original image")
        plt.subplot(1,3,2)
        plt.imshow(convd_image,cmap='gray')
        plt.title("Mine Adaptive Image")
        plt.subplot(1,3,3)
        plt.imshow(img_clahe,cmap='gray')
        plt.title("CLAHE image")
        plt.tight_layout()
        plt.savefig("images/Adaptive_image_gray.jpeg",dpi=300)
        plt.show()

        plt.subplot(1,3,1)
        plt.hist(img.flatten(),256,[0,256])
        plt.title("Original image")
        plt.subplot(1,3,2)
        plt.hist(convd_image.flatten(),256,[0,256])
        plt.title("Mine Adaptive Image")
        plt.subplot(1,3,3)
        plt.hist(img_clahe.flatten(),256,[0,256])
        plt.title("CLAHE image")
        plt.tight_layout()
        plt.savefig("images/Adaptive_image_hist_equalization.jpeg",dpi=300)
        plt.show()

    print("#"*10+"Processing Time"+"#"*10)
    print(f"Mine : {my_diff}")
    print(f"CV2 : {cv2_diff}")
    