
import numpy as np
import matplotlib.pyplot as plt
import cv2

def conv(input_image, kernel):
    # print(f"Shape of input_image : {input_image.shape}")
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
    # Get the dimensions of the input image and kernel
    
        kernel_height, kernel_width = kernel.shape
        
        # Calculate padding size for each dimension
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
       
        input_image = input_image  
        red_channel = input_image[:,:,0]
        green_channel = input_image[:,:,1]
        blue_channel = input_image[:,:,2]
      
        # Apply padding to the input image
        
        
      
        padded_input = np.pad(red_channel, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
      
            # Use sliding_window_view to green_channeleate a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
       
            # Perform element-wise multiplication and sum to get the conv result
        output=np.sum(rolling_view * kernel, axis=(-1, -2))
     
        input_image[:,:,0] = output

        #Applying green_channel channel
   
        padded_input = np.pad(green_channel, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
       
            # Use sliding_window_view to green_channeleate a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
   
            # Perform element-wise multiplication and sum to get the conv result
        output=np.sum(rolling_view * kernel, axis=(-1, -2))
       
        input_image[:,:,1] = output

        #Applying blue_channel channel
       
        padded_input = np.pad(blue_channel, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
       
            # Use sliding_window_view to create a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
      
            # Perform element-wise multiplication and sum to get the conv result
        output=np.sum(rolling_view * kernel, axis=(-1, -2))
        # print(f"Shape of output : {output.shape}")
        input_image[:,:,2] = output


        return input_image.astype(np.uint8)
    else:
        # print("GrayScale image")
        kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size for each dimension
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        
        # for i in range(num_channels):
        # Apply padding to the input image
        padded_input = np.pad(input_image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',constant_values=0)
            
            # Use sliding_window_view to create a view of the input with the same shape as the output
            
        rolling_view = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape=(kernel_height, kernel_width))
            
            # Perform element-wise multiplication and sum to get the conv result
        output=np.sum(rolling_view * kernel, axis=(-2, -1))
        # print(f"Shape of output : {output.shape}")

        return output.astype(np.uint8)

if __name__ == '__main__':
    # img = cv2.imread("Data/dog.jpeg")
    img = np.random.rand(3, 3)
    
    
    # Define the 3x3 kernel filter - BLUR FILTER
    kernel3x3 = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,1]])
    tick = cv2.getTickCount()
    convd_image = conv(img, kernel3x3)
    toc = cv2.getTickCount()
    mine_diff3 = toc - tick
    
    tick = cv2.getTickCount()
    cv2_conv_image = cv2.blur(src=img,ksize=(3,3))  # MAKING CROSS CHECK IF THE CONOLUTION IS CORRECT
    toc = cv2.getTickCount()
    cv2_diff3 = toc - tick
    cv2_conv_image_filter2d = cv2.filter2D(src=img, ddepth=-1,kernel=kernel3x3) # RECHECK IF THE conv IS CORRECT USING FILTER2D
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        plt.subplot(1,4,1)
        plt.imshow(img)
        plt.title("Original image")

        plt.subplot(1,4,2)
        plt.imshow(convd_image)
        plt.title("Mine 3x3")

        plt.subplot(1,4,3)
        plt.imshow(cv2_conv_image)
        plt.title(" Blur")
        plt.subplot(1,4,4)
        plt.imshow(cv2_conv_image_filter2d)
        plt.title(" Filter2D")
        plt.tight_layout()
        plt.savefig("images/conv_3x3.png",dpi=300)
        plt.show()
    else:
        plt.subplot(1,4,1)
        plt.imshow(img,cmap='gray')
        plt.title("Original image")

        plt.subplot(1,4,2)
        plt.imshow(convd_image,cmap='gray')
        plt.title("Mine 3x3")

        plt.subplot(1,4,3)
        plt.imshow(cv2_conv_image,cmap='gray')
        plt.title(" Blur")
        plt.subplot(1,4,4)
        plt.imshow(cv2_conv_image_filter2d,cmap='gray')
        plt.title(" Filter2D")
        plt.tight_layout()
        plt.savefig("images/gray_conv_3x3.png",dpi=300)
        plt.show()
    
    print(f"#"*10+"5x5 conv"+"#"*10)
    
    kernel5x5 = np.array([  #USED BLUR FILTER
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]
                        ])
    tick = cv2.getTickCount()
    convd_image = conv(img, kernel5x5)
    toc = cv2.getTickCount()
    mine_diff5 = toc - tick
    tick = cv2.getTickCount()
    cv2_conv_image = cv2.blur(img, ksize=(5,5)).astype(np.uint8)
    toc = cv2.getTickCount()
    cv2_diff5 = toc - tick
    cv2_conv_image_filter2d = cv2.filter2D(src=img, ddepth=-1,kernel=kernel5x5)

    img = cv2.imread("Data/dog.jpeg")
    if len(img.shape) == 3 and img.shape[2] == 3:
        plt.subplot(1,4,1)
        plt.imshow(img)
        plt.title("Original image")

        plt.subplot(1,4,2)
        plt.imshow(convd_image)
        plt.title("Mine 5x5")

        plt.subplot(1,4,3)
        plt.imshow(cv2_conv_image)
        plt.title(" Blur")
        plt.subplot(1,4,4)
        plt.imshow(cv2_conv_image_filter2d)
        plt.title(" Filter2D")
        plt.tight_layout()
        plt.savefig("images/conv_5x5.png",dpi=300)
        plt.show()
    else:
        plt.subplot(1,4,1)
        plt.imshow(img,cmap='gray')
        plt.title("Original image")

        plt.subplot(1,4,2)
        plt.imshow(convd_image,cmap='gray')
        plt.title("Mine 5x5")

        plt.subplot(1,4,3)
        plt.imshow(cv2_conv_image,cmap='gray')
        plt.title(" Blur")
        plt.subplot(1,4,4)
        plt.imshow(cv2_conv_image_filter2d,cmap='gray')
        plt.title(" Filter2D")
        plt.tight_layout()
        plt.savefig("images/gray_conv_5x5.png",dpi=300)
        plt.show()

    print("#"*10+"Proceesing Time"+ "#"*10)
    print("3x3 Kernel")
    print(f"Time - Mine: {mine_diff3/cv2.getTickFrequency()}")
    print(f"Time - CV2: {cv2_diff3/cv2.getTickFrequency()}")
    print("5x5 Kernel")
    print(f"Time - Mine: {mine_diff5/cv2.getTickFrequency()}")
    print(f"Time - CV2: {cv2_diff5/cv2.getTickFrequency()}")
    