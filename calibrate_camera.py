import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle

def calibrate_camera(): 
    # Arrays to store object points and image points from all the images.
    objpoints, imgpoints = [], [] # 3d points in real world space and 2d points in image plane.

    # Step through the list and search for chessboard corners
    for i in range(1,21):
        img = mpimg.imread('camera_cal/calibration{}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    	# prepare object points with the number of inside corners in x and y
        nx= 9
        ny = 5 if i == 1 else 6

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist        

def main():
    print('Camera Calibration processing...')
    mtx, dist = calibrate_camera()
    save_dict = {'mtx':mtx, 'dist':dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)
    print('Camera Calibration Successful!')

if __name__ == '__main__': main()
else:
    print("Successful import calibrate_camera.py!")
    main()

