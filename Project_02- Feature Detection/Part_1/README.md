# Project 2
Debashis Gupta || Graduate Student

## Part 01 : Keypoints and Detectors
<div style="text-align:justify">
Detecting the keypoints and feature descriptors of an image using any feature detection algorithms such as Harris Corner Detection, Features from Accelerated Segment Test (FAST), Scale-Invariant Feature Transform (SIFT), and Oriented Binary Feature Transform (ORB) play a significant role which allows us to use precisie information of an image for any functional task specification. Below, there is a brief description on how these algorithms effect on the different image transformations. Please note as FAST and HARRIS are keypoints detector, I have used ORB for computing the descriptors in the experimental section.
</div>

### Blurring Effect:
<div style="text-align:justify">
To get the blur effect of the original image, I have used the Gaussian blur effect on the image and the kernel size is chosen from the odd number ranging from 1 to 21. To choose the reason of this wide range of kernels is <span style="color:red"> in our class we are taught that when we use a small kernel size, the blur applied to the image is relatively mild on the other hand a larger kernel can give a more pronouced blur image relatively the earlier one. </span> Although the kernel size I have chosen is not very large, here I tried to show in a small picture of this blur effect can be found in these four algorithms. The Figure 1 shows the blur effect and the detecting keypoints drawline on the original image and the blured one.
</div>

| <center>SIFT  </center>           | <center>ORB </center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Blur/Blur_sift_7.jpg)       |   ![ORB_IMAGE](../Image/Blur/Blur_orb_7.jpg)   | 

| <center>FAST  </center>           | <center>HARRIS </center>| 
| :---------------- | :------: | 
| ![FAST](../Image/Blur/Blur_Fast_7.jpg)       |   ![HARRIS](../Image/Blur/Blur_Harris_7.jpg)   | 
<center>Figure 1: Matching Keypoints Drawline for 4 algorithms Blur Effect (kernel Size 7x7)</center>

![Blurring_Plot](../Image/Plotting/blur.jpg)
<center>Figure 2: Efficiency of the 4 Algorithms against Blur Effect</center>
<div style="text-align:justify">
The Figure 2 shows the efficicacy of these algorithms on the guassian blurring effect on the image. All the algorithms drastically drop in their precision on detecting the highly correlated keypoints between the original and transformed images with the increase of the kernel size. Hence, it concludes that with the larger blur effect these four algorithms tends to be poor in detecting keypoints. Although fast at the beginning shows promising highly correlated keypoints between the original and transformed images, it along with the other algorithms falls towards the 
</div>

### Rotation Effect: 
<div style="text-align:justify">
For the experimental purpose, I have set the rotation angle for each 45 degrees of rotation starting from 0 to 180 degrees. For each of the rotation angles, I have counted the featured-matching points of the rotated image with the original one to draw the difference how the aforementioned four algorithms (HARRIS,FAST,SIFT,and ORB) behaves in different angles. The following (Figure:3) are the images of the rotation angle for each of the algorithms with their associated identified features keypoints matching.
</div>

| <center>SIFT (180 Degree) </center>           | <center>ORB (180 Degree)</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Rotate/Rotation_sift_180.jpg)       |   ![ORB_IMAGE](../Image/Rotate/Rotation_orb_180.jpg)   | 

| <center>FAST (180 Degree) </center>           | <center>HARRIS (180 Degree)</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Rotate/Rotation_fast_180.jpg)       |   ![ORB_IMAGE](../Image/Rotate/Rotation_harris_180.jpg)   | 
<center>Figure 1: Matching Keypoints Drawline for 4 algorithms (Degree:180)</center>

![Rotation_Plot](../Image/Plotting/rotation.jpg)|
<center>Figure 4: Efficiency of the 4 Algorithms against Rotation Effect</center>

<div style="text-align:justify">
The Figure 4 shows that without any rotation FAST evaluates the most matching keypoints approximately 55000 although with the increasing in angle this algorithm falls drastically.  On the contrary, the SIFT algorithm as expected maintains the rotational effect by keeping the matching keypoints approximately same for each of the rotations. However, ORB and HARRIS both algorithms poorly perform during the changes in the rotation showing less numbers of matching keypoints between the original image and the rotated image.
</div>


### Scale Effect:
<div style="text-align:justify"> 
For checking the scale effect I set up the same image as before with a variation starting from half of the size of the actual image to the dobule of the size of the image. Here, I tried to find how these four algorithms can perform to this scale transformation on matching the keypoints between the original image and the scaled image. The following Figure 5 shows an example of 1.5 times scaled image of the original one.
</div>

| <center>SIFT (1.5x) </center>           | <center>ORB (1.5x)</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Scale/Scale_sift_1.5.jpg)       |   ![ORB_IMAGE](../Image/Scale/Scale_orb_1.5.jpg)   | 

| <center>FAST (1.5x) </center>           | <center>HARRIS (1.5x)</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Scale/Scale_fast_1.5.jpg)       |   ![ORB_IMAGE](../Image/Scale/Scale_Harris_1.5.jpg)   | 
<center>Figure 5: Matching Keypoints Drawline for 4 algorithms (Scale : 1.5x)</center>

![Scale_Plot](../Image/Plotting/scale.jpg)|
<center>Figure 6: Efficiency of the 4 Algorithms against Scale Effect</center>
<div style="text-align:justify">
The figure 6 shows that SIFT depicts some consistency in detecting the keypoints and matching them between original and scaled image having around 12000 matching points varying from the original shape upto the double scale. However, when the image is scaled down to its half of original scale SIFT reduces to approximately 5000 keypoints. On the contrary, FAST keypoint detectors, on the original scale, provides the best matching point although it falls down when the image is scaled down to its half or scaled up to the double of original scale. Hence, this falling trend shows that FAST keypoint detectors can not detect opitimal keypoints during scaling up the image. Finally, Harris corner detection technique can detect much better keypoints on the original scale of the given image than ORB however, the performance falls down and almost went to the identical as ORB curve shows during the scale up and scale down of the original image.
</div>

### Illumination Effect: 
<div style="text-align:justify">
To check the robustness of these four algorithms agianst the Illumination more particularly when it is low light comparision I keep the values symmetic ranging from negative to positive values. The figure 7 illustrates the performance of matching drawlines for these four algorithms between the original image and with 30% light reduction effect of the original image.
</div>

| <center>SIFT </center>           | <center>ORB</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Light/Light_sift_0.7.jpg)       |   ![ORB_IMAGE](../Image/Light/Light_orb_0.7.jpg)   | 

| <center>FAST </center>           | <center>HARRIS</center>| 
| :---------------- | :------: | 
| ![SIFT_IMAGE](../Image/Light/Light_fast_0.7.jpg)       |   ![ORB_IMAGE](../Image/Light/Light_Harris_0.7.jpg)   | 
<center>Figure 7: Matching Keypoints Drawline for 4 algorithms (Light : 70% of original)</center>

![Light_Plot](../Image/Plotting/low_light_illumination.jpg)|
<center>Figure 8: Efficiency of the 4 Algorithms against Illumination Effect</center>

<div style="text-align:justify">
As the figure 8 describes FAST keypoints detection techniques works well for the image that has close intensity values (90%) to the original image whereas it falls drastically when the image has become 10% intensity values of the original image. However, the rest of the algorithms show a consistency in detecting the keypoints of the image although have a minor deviation in detection while the light of the original image is decreasing from its 100% to 10%. 
</div>


## Summary of Matching Keypoints for 4 algorithms

The following tables describes the number of matching keypoints for SIF, ORB, Fast and Harris. 

| Rotation Angles | 0     | 45   | 90   | 135   | 180   |
|-----------------|-------|------|------|-------|-------|
| SIFT            | 11612 | 7218 | 7342 | 10390 | 11598 |
| ORB             | 500   | 347  | 344  | 337   | 411   |
| FAST            | 55853 | 3758 | 5551 | 4210  | 8733  |
| HARRIS          | 2517  | 161  | 176  | 49    | 424   |

| Scale  | 0.5  | 1.0   | 1.5   | 2.0   |
|--------|------|-------|-------|-------|
| SIFT   | 4943 | 11612 | 10232 | 10116 |
| ORB    | 183  | 500   | 278   | 197   |
| FAST   | 3898 | 55853 | 5276  | 2995  |
| HARRIS | 166  | 2517  | 422   | 422   |

| Illumination | 10%   | 30%  | 70%   | 90%   |
|--------------|-------|------|-------|-------|
| SIFT         | 10038 | 6710 | 6879  | 10127 |
| ORB          | 117   | 121  | 470   | 470   |
| FAST         | 6834  | 4332 | 16276 | 37145 |
| HARRIS       | 38    | 7    | 1895  | 10127 |

| Blur   | 1x1   | 7x7  | 15x15 | 21x21 |
|--------|-------|------|-------|-------|
| SIFT   | 11612 | 5147 | 2110  | 1581  |
| ORB    | 500   | 316  | 172   | 100   |
| FAST   | 55853 | 2896 | 224   | 20    |
| HARRIS | 2517  | 912  | 439   | 346   |

<span style="color:blue"> [Only showing some values for blurring]</span>


## Analysis of the Invariances Property of these Algorithms
<div style="text-align:justify">
In theroy, the Invariances Property means that the algorithm should be consistent with their detecting the keypoints irespective to the transforms made on the image such as translation of the scale, rotation, blurring or changing the intensity levels of the image. However, the tables shown above, clearly depict that these algorithms i.e. SIFT, ORB, FAST and Harris do not show this invariance property as they all are prone to the changes made into the transformed images. Although SIFT and FAST can detect much more keypoints than HARRIS and particularly ORB, they also drastically fall when the image is transformed using either rotation, scale, blur or illumination. However, <span style="color:red">I found that although the ORB detects much lower keypoints than the other algorithms, it showed some consistency in detecting those keypoints again and again irrespective to the changes made in transforming the original image which clearly make it more suitable in terms of invariance. </span>
</div>
