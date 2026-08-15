import cv2
import numpy as np
from simple.matcher import Matcher

class Stitcher:
    """
    Stitcher class:
    
    """
    def __init__(self, featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres):
        """Initialize Stitcher object

        Args:
            featureExtractor (int): feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB, 4: SIFT, 5: SURF (*))
            matchingMethod (int): homography matching method (0: least-square method, 1: RANSAC method, 2: Least-Median robust method, 3: PROSAC-based robust method)
            retentionThres (float): feature retention threshold.
            reprojThresh (int): reprojection threshold for homography estimation.
            matchThres (int): number of matches threshold.
        """        
        # matcher obj
        self.matcher = Matcher(featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres)
                        
        # cache homography matrices
        self.cacheHs = None
        
        # cache warped image sizes
        self.cacheSs = None
        
        # cache offsets
        self.cacheOs = None
        
        # cache number of images
        self.cacheN = None
        
    def stitch(self, imgs):
        """Stitch images

        Args:
            imgs (_type_): input images

        Returns:
            retval: True if stitching successful and otherwise
            result: stitching result (None if the stitching process is failed)
        """        
        if self.cacheN is None:
            self.cacheN = len(imgs) - 1
            self.cacheHs = [None] * self.cacheN
            self.cacheSs = [None] * self.cacheN
            self.cacheOs = [None] * self.cacheN
        
        result = None
        
        for i in range(self.cacheN):
            if i == 0:
                trainImg = imgs[i + 1]
                queryImg = imgs[i]
            else:
                if result is None:
                    return False, None
                trainImg = result
                queryImg = imgs[i + 1]
            if self.cacheHs[i] is None:
                # if os.path.exists("cacheH_{}.npy".format(i)):
                #     self.cacheHs[i] = np.load("cacheH_{}.npy".format(i))
                #     self.cacheSs[i] = np.load("cacheS_{}.npy".format(i))
                #     self.cacheOs[i] = np.load("cacheO_{}.npy".format(i))
                # else:
                self.cacheHs[i], self.cacheSs[i], self.cacheOs[i] = self.matcher.generateHomography(trainImg, queryImg)
            
            if self.cacheHs[i] is not None:
                # if not os.path.exists("cacheH_{}.npy".format(i)):
                #     np.save("cacheH_{}.npy".format(i), self.cacheHs[i])
                #     np.save("cacheS_{}.npy".format(i), self.cacheSs[i])
                #     np.save("cacheO_{}.npy".format(i), self.cacheOs[i])
                tmp = cv2.warpPerspective(queryImg, self.cacheHs[i], self.cacheSs[i], borderMode=cv2.BORDER_TRANSPARENT)
                tmp[self.cacheOs[i][0] : trainImg.shape[0] + self.cacheOs[i][0], self.cacheOs[i][1] : trainImg.shape[1] + self.cacheOs[i][1]] = trainImg
                result = tmp
                self.cacheHs[i] = None
            else:
                print("None: ", i)
                return False, None
                
        return True, result


        