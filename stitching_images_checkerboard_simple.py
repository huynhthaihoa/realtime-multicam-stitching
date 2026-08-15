import os

import cv2
from checkerboard_simple.stitcher import Stitcher as scb
from simple.stitcher import Stitcher as ss
from advanced.stitcher import Stitcher as sa
from glob import glob
# from checkerboard_simple.utils import Calibrator

stitcher_checkerboard = scb(1, 10)#8, 5) # use checkerboard corners as feature (simple version)
stitcher_simple = ss(4, 1, -1, 4, 10) # use image keypoints as feature (simple version)
stitcher_advanced = sa(feature_type = 4) # use image keypoints as feature (advanced version)
leftPaths = glob("D:\\SOURCE\\synchronization\\2023_03_30_10_24_03\\C\\*.jpg") # images from the left camera

for leftPath in leftPaths:
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):  # q to quit
        break
    name = os.path.basename(leftPath)
    rightPath = f"D:\\SOURCE\\synchronization\\2023_03_30_10_24_03\\D\\{name}" # images from the right camera
    frontPath = f"D:\\SOURCE\\synchronization\\2023_03_30_10_24_03\\B\\{name}" # images from the front camera
    frontImg = cv2.imread(frontPath)
    leftImg = cv2.imread(leftPath)
    rightImg = cv2.imread(rightPath)
    imgs = [leftImg, frontImg, rightImg]
    success, stitched = stitcher_checkerboard.stitch(imgs)
    if not success:
        print("Fail!")
        continue
    else:
        print("Success!")
    stitched = cv2.resize(stitched, (1920, 1080))
    cv2.imshow("Stitching", stitched)
    cv2.imwrite(f"RESULT\\CHECKERBOARD\\{name}", stitched)
cv2.destroyAllWindows()
    
# trainPaths = glob("D:\\SOURCE\\synchronization\\BIG\\AC\\0\\*.jpg")
