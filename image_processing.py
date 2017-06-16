import cv2
import numpy as np
from helper import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) # Rescale back to 8 bit integer
    binary_output = np.zeros_like(scaled_sobel) # Create a copy and apply the threshold
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def magnitude_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2) # Calculate the gradient magnitude
    scale_factor = np.max(gradmag)/255 # Rescale to 8 bit
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_threshold(img, thresh=(100, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def lab_b_threshold(img, thresh=(0, 255)):
    # Thresholds the B-channel of LAB
    lab_b = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2]
    binary_lab_b = np.zeros_like(lab_b)
    binary_lab_b[(lab_b > thresh[0]) & (lab_b <= thresh[1])] = 1
    return binary_lab_b

def combined_threshold(img):
    abs_x_bin = abs_sobel_threshold(img, orient='x', thresh_min=20, thresh_max=255)
    mag_bin = magnitude_threshold(img, sobel_kernel=5, mag_thresh=(10, 255))
    dir_bin = direction_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_bin = hls_threshold(img, thresh=(170, 255))
    yv_bin = yuv_threshold(img,y_thresh=(200,255),u_thresh=(200,255),v_thresh=(0,100))
    lab_b_bin = lab_b_threshold(img, thresh
=(140,255))
    mag_dir = np.zeros_like(dir_bin)
    mag_dir[(mag_bin == 1) & (dir_bin == 1)] = 1
    combined = np.zeros_like(dir_bin)
    combined[(abs_x_bin == 1)|(mag_dir==1)|(lab_b_bin==1)|(hls_bin == 1)] = 1
    return combined,abs_x_bin, mag_bin, dir_bin, hls_bin, lab_b_bin

if __name__ == '__main__':
    img = mpimg.imread('test_images/straight_lines1.jpg')
    h,w,c = img.shape
    combined,abs_x_bin, mag_bin, dir_bin, hls_bin, lab_b_bin = combined_threshold(img)

    for i, img in enumerate(['combined','abs_x_bin', 'mag_bin', 'dir_bin', 'hls_bin', 'lab_b_bin']):
        showimg((3, 2, i + 1), img, eval(img))
    plt.show()

    combined, M_warp, Minv_warp = warp_image(combined, h, w)
    abs_x_bin, M_warp, Minv_warp = warp_image(abs_x_bin, h, w)
    mag_bin, M_warp, Minv_warp = warp_image(mag_bin, h, w)
    dir_bin, M_warp, Minv_warp = warp_image(dir_bin, h, w)
    hls_bin, M_warp, Minv_warp = warp_image(hls_bin, h, w)
    lab_b_bin, M_warp, Minv_warp = warp_image(lab_b_bin, h, w)

    for i, img in enumerate(['combined','abs_x_bin', 'mag_bin', 'dir_bin', 'hls_bin', 'lab_b_bin']):
        showimg((3, 2, i + 1), img, eval(img))
    plt.show()

