import cv2
import numpy as np

class Matcher:
    """
    Matcher class
    """
    
    def __init__(self, featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres):
        """Initialize Matcher object

        Args:
            featureExtractor (int): feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB, 4: SIFT, 5: SURF (*))
            matchingMethod (int): homography matching method (0: least-square method, 1: RANSAC method, 2: Least-Median robust method, 3: PROSAC-based robust method)
            retentionThres (float): feature retention threshold.
            reprojThresh (int): reprojection threshold for homography estimation.
            matchThres (int): number of matches threshold.
        """                
        if featureExtractor == 0:
            self.extractor = cv2.AKAZE_create()
        elif featureExtractor == 1:
            self.extractor = cv2.BRISK_create()
        elif featureExtractor == 2:
            self.extractor = cv2.KAZE_create()
        elif featureExtractor == 3:
            self.extractor = cv2.ORB.create()
        elif featureExtractor == 5:
            self.extractor = cv2.xfeatures2d_SURF.create()
        else:
            self.extractor = cv2.SIFT_create()
        
        if matchingMethod == 0:
            self.matchingMethod = 0
        elif matchingMethod == 3:
            self.matchingMethod = cv2.RHO
        elif matchingMethod == 2:
            self.matchingMethod = cv2.LMEDS
        else:
            self.matchingMethod = cv2.RANSAC
        
        self.retentionThres = retentionThres
        
        if self.retentionThres == -1:
            if featureExtractor == 3: # ORB
                self.retentionThres = 0.3
            else: # others
                self.retentionThres = 0.65
        
        self.reprojThresh = reprojThresh
        
        self.matchThres = matchThres
        
        if featureExtractor < 4:
            index_params= dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) #2
        else:
            index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    def getFeatures(self, img):
        """Get features from input image

        Args:
            img: input image

        Returns:
            features as a dictionary (keypoint & description)
        """        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.extractor.detectAndCompute(gray, None)
    
    def generateHomography(self, trainImg, queryImg):
        """Generate homography matrix between two images

        Args:
            trainImg: train image
            queryImg: query image (image to be perspective transformed)

        Returns:
            H_sum: homography matrix to perspectively transform queryImg plane to trainImg plane
            (x_max - x_min, y_max - y_min): warped image size
            (x_min, y_min): offset for placing trainImg into the warped image
        """   
        trainKps, trainDes = self.getFeatures(trainImg)
        queryKps, queryDes = self.getFeatures(queryImg)
        
        # Each keypoint of the first image is matched with a number of keypoints from the second image. 
        # We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). 
        # Lowe's test checks that the two distances are sufficiently different. 
        # If they are not, then the keypoint is eliminated and will not be used for further calculations.
        matches = self.matcher.knnMatch(queryDes, trainDes, k=2)
        
        good = list()
        for match in matches:
            if len(match) < 2:
                continue
            if match[0].distance < match[1].distance * self.retentionThres:
                good.append(match[0])    
        
        # Extract location of good matches
        # Computing a homography between two sets of points requires at a bare minimum an initial set of 4 matches. 
        # For a more reliabe estimation, we should have substantially more than just 4 matched points.
        if len(good) > self.matchThres:

            queryMatchedPts = np.array([queryKps[good_match.queryIdx].pt for good_match in good], dtype = np.float32)
            queryMatchedPts.reshape((-1, 1, 2))
            
            trainMatchedPts = np.array([trainKps[good_match.trainIdx].pt for good_match in good], dtype = np.float32)
            trainMatchedPts.reshape((-1, 1, 2))

            # Find homography
            H, _ = cv2.findHomography(
                queryMatchedPts, trainMatchedPts, self.matchingMethod, self.reprojThresh
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
            
            H_sum = H_translation.dot(H)
            
            return H_sum, (x_max - x_min, y_max - y_min), (-y_min, -x_min)
        return None, None, None

