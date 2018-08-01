# You should replace these 3 lines with the output in calibration step
import cv2
import numpy as np
import sys

DIM=(640, 480)
K=np.array([[333.3095682593701, 0.0, 299.42451764331906], [0.0, 333.39606366460384, 227.39887052277433], [0.0, 0.0, 1.0]])
D=np.array([[-0.057478717587506466], [0.10222798530534297], [-0.13515780465585153], [0.05446853521315723]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)