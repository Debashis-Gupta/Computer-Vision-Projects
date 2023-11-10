import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image if shows if any exception occurs

def fun_filter(input_image):

    if input_image is None:
        print(f"Error: Could not load the image...")
    else:
        # Apply point operations (e.g., adjust brightness and contrast)
        brightness = 1.5
        contrast = 5
        adjusted_image = cv2.convertScaleAbs(input_image, alpha=contrast, beta=brightness)

        # Apply a Gaussian blur filter
        blurred_image = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

        # Apply edge detection using Canny
        edges = cv2.Canny(blurred_image, 10, 20)

        # Create a mask for pixelation (you can adjust the block size)
        block_size = 20
        masked_image = cv2.resize(input_image, None, fx=1/block_size, fy=1/block_size, interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(masked_image, input_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Combine the filtered images using bitwise operations
        final_image = cv2.bitwise_or(adjusted_image, adjusted_image, mask=edges)
        final_image = cv2.bitwise_and(final_image, pixelated_image)

        # Display the original and filtered images using Matplotlib
        plt.figure(figsize=(10, 5))
        
         # Fun Image Filter
        # plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        plt.title('Guess Image ')
        plt.savefig("images/Guess_Image.jpeg",dpi=300)
        plt.show()
        # Original image
        # plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.savefig("images/Original_Image.jpeg",dpi=300)
        plt.show()
       


if __name__ == "__main__":
    # Load the input image
    input_image = cv2.imread('Data/Dog.jpeg')
    fun_filter(input_image)

