# Canny-Edge-Detector

To create myCannyEdgeDetector(image, Low_Threshold, High_Threshold) function following steps were taken into consideration:
1.	 First, the image was converted from RGB to grayscale.
2.	 Gaussian filter was created to smoothen the image, and reduce noise.
3.	 Using sobel filter, the magnitude and orientation of the gradient were calculated.
4.	 Computed non maximal suppression to make sure that we consider only those pixels who contribute strongly to the edge.
5.	 Performed (double) Thresholding to strengthen those pixels who have high intensity so that they contribute to the final edge and weaken those pixels who have low intensity so that they can be ignored. High threshold is used to identify strong pixels and low threshold is used to identify pixels which are not relevant
6.	 Performed hysteresis to transform those weak pixels to strong which have at least one of the pixels around them which is strong to make sure we don’t miss pixels which contribute to the edge detection.
7.	 Calculated Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Metric (SSIM) between skimage canny edge detector output and myCannyEdgeDetector() output (printed both in terminal).

To decide Low_Threshold and High_Threshold, I have considered and observed on the basis of standard values and applied trial and error method to get a hold of the best possible combination of threshold values.
As we decrease High_Threshold value (weak), more small minor edges will also be considered or detected, and if we increase the High_Threshold value (strong), less edges are considered or detected (vice versa with Low_Threshold value).
