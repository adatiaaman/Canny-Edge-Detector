# Aman Pankaj Adatia
# 2020CSB1154
# Task 1

from math import log10, sqrt
from skimage import feature 
import skimage.filters as filters
from skimage.color import *
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import skimage.io as io
from skimage.metrics import structural_similarity as ssim

# (i) output of skimage.feature.canny for the input image

inputImg = io.imread('image/img_1.jpg')
inputImg = rgb2gray(inputImg)
gausImg = filters.gaussian(inputImg, sigma=1)
skCanny = feature.canny(gausImg)
plt.imsave('output/sk_canny.jpg', skCanny, cmap='gray')

# (ii) edge output of myCannyEdgeDetector()

# function to calculate convolution between image and filter (mask)
def convolution(image, mask2d):
        result = np.zeros(image.shape)

        h, w = image.shape
        mask_h, mask_w = mask2d.shape
        # padding - adding pixels to image for convinience of computation
        pad_h = int((mask_h-1)/2)
        pad_w = int((mask_w-1)/2)
        pad_img = np.zeros((h + (2*pad_h), w+(2*pad_w)))
        pad_img[pad_h:pad_img.shape[0]-pad_h, pad_w:pad_img.shape[1]-pad_w] = image

        for i in range(h):
                for j in range(w):
                        conv1 = mask2d
                        conv2 = pad_img[i:i+mask_h, j:j+mask_w]
                        # convolution formula
                        result[i, j]=np.sum(conv1*conv2) # scan over every window
        
        return result

def gaussianFilter(image):
        mask1d = np.linspace(-(3//2), 3//2, 3) # array of equally spaced numbers in provided range
        sigma = 1 # variance set to 1 always in this case (can be changed or taken as function parameter)

        for i in range(3):
                # applying gaussian function - normal distribution
                mask1d[i]=1/(np.sqrt(2*np.pi)*sigma)*(np.e**(-np.power(mask1d[i]/sigma, 2)/2))
        # converting to 2D
        mask2d = np.outer(mask1d.T, mask1d.T)
        mask2d *= 1.0/mask2d.max()

        return convolution(image, mask2d)

# function to apply sobel filter
def sobelFilter(image):
        # assorted finite difference filter
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Mask in X
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # Mask in Y

        im_x = convolution(image, sobel_x) # applying horizontal mask - Gx
        im_y = convolution(image, sobel_y) # applying vertical mask - Gy

        grad_sq = np.square(im_x)+np.square(im_y)
        gradient = np.sqrt(grad_sq) # G=sqrt(Gx^2+Gy^2)
        magnitude = gradient*255/gradient.max() # normalize output between 0 to 255 - magnitude
        theta = np.arctan2(im_y, im_x)*180/np.pi # slope of the gradient - converting to degree from radian

        return (magnitude, theta)

# function to perform non maximum suppression to thin out the edges
def nonMaxSuppression(gradientMagnitude, theta):
        h, w = gradientMagnitude.shape
        nms = np.zeros(gradientMagnitude.shape)

        for i in range(1, h-1):
                for j in range(1, w-1):
                        orientation = theta[i, j]
                        checker = 180.0/8 # to check quadrant wise
                        before=0
                        after=0
                        # identifying edge direction based on theta matrix obtained from sobel
                        if (0<=orientation<checker) or (15*checker<=orientation<=16*checker):
                                before = gradientMagnitude[i, j-1]
                                after = gradientMagnitude[i, j+1]
                        elif (checker<=orientation<3*checker) or (9*checker<=orientation<=11*checker):
                                before = gradientMagnitude[i+1, j-1]
                                after = gradientMagnitude[i-1, j+1]
                        elif (3*checker<=orientation<5*checker) or (11*checker<=orientation<=13*checker):
                                before = gradientMagnitude[i-1, j]
                                after = gradientMagnitude[i+1, j]
                        else:
                                before = gradientMagnitude[i-1, j-1]
                                after = gradientMagnitude[i+1, j+1]
                        
                        # points that definitely lie on edges - points that are maxima
                        if (gradientMagnitude[i, j]>=before) and (gradientMagnitude[i, j]>=after):
                                nms[i, j]=gradientMagnitude[i, j]

        return nms

# function to perform thresholding
def threshold(image, Low_Threshold, High_Threshold):
        result = np.zeros(image.shape) # contains points whose gradient magnitude is greater than high threshold

        high = image.max()*High_Threshold
        low = image.max()*(High_Threshold*Low_Threshold)
        strong = 255
        weak = 35

        # strong pixels have high intensity so that they contribute to the final edge
        strong_row, strong_col = np.where(image>=high)
        result[strong_row, strong_col] = strong
        # weak pixels have low intensity, not to be considered for the final edge detection
        weak_row, weak_col = np.where((image<=high) & (image>=low))
        result[weak_row, weak_col] = weak

        return result

# function to apply hysteresis
def hysteresis(img, weak=45, strong=255):
        # hysteresis - transforming weak pixels into strong 
        # if and only if at least one of the pixels around the one being processed is strong
        image = img.copy()
        h, w = image.shape

        neighbour = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for i in range(1, h-1):
                for j in range(1, w-1):
                        if image[i, j] == weak:
                                found=0
                                for k in neighbour: # loop to check neighbourhood
                                        ii = i+k[0]
                                        jj = j+k[1]
                                        if image[ii, jj] == strong: # found strong neighbour
                                                found=1
                                if found == 1: 
                                        image[i, j]=strong
                                else:
                                        image[i, j]=0
        return image


def PSNR(img1, img2):
        mse = np.mean(np.abs(img1-img2)**2)
        if mse != 0: 
                psnr = log10(255.0/sqrt(mse))
                psnr *= 20 # dB
                return psnr
        else:
                return 100 # max

def myCannyEdgeDetector(image, Low_Threshold=0.05, High_Threshold=0.12):
        # applying each function one by one
        gaus = gaussianFilter(image) # step 1
        sobel, direction = sobelFilter(gaus) # step 2
        nms = nonMaxSuppression(sobel, direction) # step 3
        thresholdImg = threshold(nms, Low_Threshold, High_Threshold) # step 4
        hyst = hysteresis(thresholdImg, 35, 255) # step 5
        
        plt.imsave('output/my_canny.jpg', hyst, cmap = 'gray')

        # print output at each step
        # fig, axes = plt.subplots(1, ncols=6, figsize=(18, 9))
        # axes[0].imshow(image, cmap = 'gray')
        # axes[0].set_title('Original image')
        # axes[0].axis('off')
        # axes[1].imshow(gaus, cmap = 'gray')
        # axes[1].set_title('Gaussian Image')
        # axes[1].axis('off')
        # axes[2].imshow(sobel, cmap = 'gray')
        # axes[2].set_title('Sobel Image')
        # axes[2].axis('off')
        # axes[3].imshow(nms, cmap = 'gray')
        # axes[3].set_title('NonMaxSup Image')
        # axes[3].axis('off')
        # axes[4].imshow(thresholdImg, cmap = 'gray')
        # axes[4].set_title('Threshold Image')
        # axes[4].axis('off')
        # axes[5].imshow(hyst, cmap = 'gray')
        # axes[5].set_title('Hysteresis Image')
        # axes[5].axis('off')
        # plt.show()
        return hyst


img = io.imread('image/img_1.jpg')
imgGray = rgb2gray(img)
myCanny = myCannyEdgeDetector(imgGray, 0.05, 0.12)

# (iii) Compute the peak signal to noise ratio (PSNR) and Structural Similarity Index Metric (SSIM) 
# between the outputs of skimage’s canny edge detector and the myCannyEdgeDetector(). 
# Display the PSNR and SSIM values too

psnr_value = PSNR(skCanny, myCanny) 
ssim_value = ssim(skCanny.astype(bool), myCanny.astype(bool))
text = f'PSNR (Peak Signal to Noise Ratio) value = {psnr_value} \nSSIM (Structural Similarity Index Metric) value = {ssim_value}'

# print and compare outputs
fig, axes = plt.subplots(1, ncols=3, figsize=(19, 8))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(skCanny, cmap = 'gray')
axes[1].set_title('skimage.feature.canny output')
axes[1].axis('off')
axes[2].imshow(myCanny, cmap = 'gray')
axes[2].set_title('myCannyEdgeDetector() output')
axes[2].axis('off')
plt.figtext(0.15, 0.25, text)
plt.show()


print(f"PSNR value = {psnr_value}") # closer to 100, more efficient
print(f"SSIM value = {ssim_value}") # closer to 1, better structural similarity

