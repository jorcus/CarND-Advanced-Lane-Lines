from moviepy.editor import VideoFileClip
from line import Line
from helper import *
from image_processing import combined_threshold

def pipeline(img):
    h, w, c = img.shape
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img = gaussian_blur(undist, kernel=5) #undistorted image
    combined,abs_x_bin, mag_bin, dir_bin, hls_bin, lab_b_bin = combined_threshold(img) # Combined all threshold
    warped,M_warp,Minv_warp = warp_image(combined,h,w)
    # Perspective Transform into Bird-eye View
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = finding_lines(warped, nonzeroy, nonzerox)
    
    out_img = cv2.warpPerspective(out_img, Minv_warp, (w, h)) # Rectangle roadlines
    #window_centroids = find_window_centroids(warped)
    #windows_result = window_centroids_logits(window_centroids,window_width, window_height, warped)
    line_highlight ,left_fit,right_fit,left_lane_inds,right_lane_inds,window_img = finding_lines2(warped, left_fit, right_fit, out_img,nonzeroy,nonzerox)
    
    out_lines = cv2.warpPerspective(window_img, Minv_warp, (w, h))
    result = final_result(undist, left_fit, right_fit, Minv_warp,left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
    result = cv2.addWeighted(result, 1, out_lines, 1, 0)
    result = cv2.addWeighted(result, 1, out_img, .3, 0)
    return result   

if __name__ == "__main__":
    print("Image line detecting...")
###
#The printing below little bit messy, will create an loop function later once finish the term.
###
    img = cv2.imread('test_images/img0410.png')
    img2 = cv2.imread('test_images/img0411.png')
    img3 = cv2.imread('test_images/img0412.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0413.png')
    img2 = cv2.imread('test_images/img0414.png')
    img3 = cv2.imread('test_images/img0415.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0416.png')
    img2 = cv2.imread('test_images/img0417.png')
    img3 = cv2.imread('test_images/img0418.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0419.png')
    img2 = cv2.imread('test_images/img0420.png')
    img3 = cv2.imread('test_images/img0421.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0422.png')
    img2 = cv2.imread('test_images/img0423.png')
    img3 = cv2.imread('test_images/img0424.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')


    img = cv2.imread('test_images/img0210.png')
    img2 = cv2.imread('test_images/img0211.png')
    img3 = cv2.imread('test_images/img0212.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0213.png')
    img2 = cv2.imread('test_images/img0214.png')
    img3 = cv2.imread('test_images/img0215.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0216.png')
    img2 = cv2.imread('test_images/img0217.png')
    img3 = cv2.imread('test_images/img0218.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0219.png')
    img2 = cv2.imread('test_images/img0220.png')
    img3 = cv2.imread('test_images/img0221.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/img0222.png')
    img2 = cv2.imread('test_images/img0223.png')
    img3 = cv2.imread('test_images/img0224.png')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/test1.jpg')
    img2 = cv2.imread('test_images/test2.jpg')
    img3 = cv2.imread('test_images/test3.jpg')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')


    img = cv2.imread('test_images/test4.jpg')
    img2 = cv2.imread('test_images/test5.jpg')
    img3 = cv2.imread('test_images/test6.jpg')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')

    img = cv2.imread('test_images/straight_lines1.jpg')
    img2 = cv2.imread('test_images/straight_lines2.jpg')
    img3 = cv2.imread('test_images/straight_lines1.jpg')

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30,10))
    ax1.imshow(img)
    ax1.axis('off')
    ax2.imshow(pipeline(img))
    ax2.axis('off')
    ax3.imshow(img2)
    ax3.axis('off')
    ax4.imshow(pipeline(img2))
    ax4.axis('off')
    ax5.imshow(img3)
    ax5.axis('off')
    ax6.imshow(pipeline(img3))
    ax6.axis('off')
    plt.show()
