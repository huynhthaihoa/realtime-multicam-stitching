import cv2
import numpy as np
from .utils import CornerDetector
class Stitcher:
    """
    Stitcher class:
    
    """
    def __init__(self, method=0, threshold=3):
        """Initialize Stitcher object

        Args:
            method (int): method used to compute a homography matrix (0: least squares method - default, 1: RANSAC, 2: LMEDS, 3: RHO)
            threshold (double): maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only). Default value is 3.
        """        
        # initialize checkerboard detectors
        detector0 = CornerDetector((11, 4))
        detector1 = CornerDetector((8, 5))
        self.detectors = [detector0, detector1]
        
        # matcher method
        self.method = method
        if method == 1:
            self.method = cv2.RANSAC
        elif method == 2:
            self.method = cv2.LMEDS
        elif method == 3:
            self.method = cv2.RHO
        self.threshold = threshold
                             
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
            imgs: input images (from left to right)

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
            # Warp the queryImg to the same plane with trainImg
            if i == 0:
                trainImg = imgs[i + 1]
                queryImg = imgs[i]
            else:
                if result is None:
                    return False, None
                trainImg = result
                queryImg = imgs[i + 1]

            if self.cacheHs[i] is None:
                
                # Detect checkerboard corners from trainImg (destination) and queryImg (source)
                if i == 0:
                    querySuccess, queryCorners = self.detectors[0].getCorners(queryImg)
                    trainSuccess, trainCorners = self.detectors[0].getCorners(trainImg)
                    cacheImage = trainImg.copy()
                else:
                    querySuccess, queryCorners = self.detectors[i % 2].getCorners(queryImg)
                    trainSuccess, trainCorners = self.detectors[i % 2].getCorners(cacheImage)
                    cacheImage = queryImg.copy()
                                 
                if not (querySuccess and trainSuccess):
                    return False, None
                
                if i != 0:
                    updateCorners = list()
                    for corner in trainCorners:
                        corner = np.array(corner)
                        corner = np.squeeze(corner)
                        new_corner = np.array((corner[0] + self.cacheOs[i - 1][1], corner[1] + self.cacheOs[i - 1][0]))
                        cv2.circle(trainImg, new_corner.astype(np.int32), 3, (0, 255, 0), 1)
                        new_corner = np.expand_dims(new_corner, axis=0)
                        updateCorners.append(new_corner)
                    trainCorners = np.array(updateCorners)
                else:
                    cv2.drawChessboardCorners(trainImg, self.detectors[i % 2].getCheckerboardSize(), trainCorners, trainSuccess)

                # Find initial homography
                H, _ = cv2.findHomography(
                    queryCorners, trainCorners, self.method, self.threshold
                )
                
                points0 = np.array([[0, 0], [0, trainImg.shape[0]], [trainImg.shape[1], trainImg.shape[0]], [trainImg.shape[1], 0]], dtype = np.float32)
                points0 = points0.reshape((-1, 1, 2))
                points1 = np.array([[0, 0], [0, queryImg.shape[0]], [queryImg.shape[1], queryImg.shape[0]], [queryImg.shape[1], 0]], dtype = np.float32)
                points1 = points1.reshape((-1, 1, 2))
                points2 = cv2.perspectiveTransform(points1, H)                
                points = np.concatenate((points0, points2), axis=0)
                
                [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(np.int32)
                [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(np.int32)
                
                H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
                
                self.cacheHs[i] = H_translation.dot(H) # final homography matrix     
                self.cacheSs[i] = (x_max - x_min, y_max - y_min) # warped image size
                self.cacheOs[i] = (-y_min, -x_min) # position to put trainImg to warped result
                
            if self.cacheHs[i] is not None:
                tmp = cv2.warpPerspective(queryImg, self.cacheHs[i], self.cacheSs[i], borderMode=cv2.BORDER_TRANSPARENT)
                tmp[self.cacheOs[i][0] : trainImg.shape[0] + self.cacheOs[i][0], self.cacheOs[i][1] : trainImg.shape[1] + self.cacheOs[i][1]] = trainImg
                result = tmp
                self.cacheHs[i] = None
            else:
                return False, None
                
        return True, result


        