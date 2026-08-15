import cv2
import numpy as np
# import argparse
from math import sqrt, ceil

RGB_CORNER_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE #| cv2.CALIB_CB_FAST_CHECK
SUBPIX_CRITERIA = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1

def _pdist(p1, p2):
    """
    L-2 distance of two points
    
    Args:
        p1 = (x1, y1)
        p2 = (x2, y2)    
    """
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

class CornerDetector:
    """Corner Detector class:
    
    Properties:
        img_shape: image shape
        nRows: checkerboard's number of rows
        nCols: checkerboard's number of columns
        nBorders: number of border pixels
        is_low_res: working on low resolution (True) or high resolution (False) stream 
    """    
    def __init__(self, checkerboard, is_low_res = True, border = 8):
        """Instance initialization

        Args:
            checkerboard (tuple): checkerboard shape (number of rows * number of columns)
            is_low_res (bool): working on low resolution (True) or high resolution (False) stream 
            border (int): number of border pixels (default is 8)
        """        
        self.img_shape = None
        
        # Make sure n_cols > n_rows to agree with OpenCV CB detector output
        self.nCols = max(checkerboard[0], checkerboard[1])
        self.nRows = min(checkerboard[0], checkerboard[1])
        
        self.nBorders = border
        self.is_low_res = is_low_res
        
    def getCheckerboardSize(self):
        """Get checkerboard size (tuple of (nRows, nCols))
        """        
        return (self.nCols, self.nRows)
            
    def getOutsideCorners(self, corners):
        """Return the four corners of the board as a whole, as (up_left, up_right, down_right, down_left).

        Args:
            corners (list): input corners

        Raises:
            Exception: Invalid number of corners

        Returns:
            (up_left, up_right, down_right, down_left)
        """        
        if corners.shape[1] * corners.shape[0] != self.nCols * self.nRows:
            raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" % (corners.shape[1] * corners.shape[0],
                                                                                self.nCols, self.nRows))
        up_left    = corners[0, 0]
        up_right   = corners[self.nCols - 1, 0]
        down_right = corners[-1, 0]
        down_left  = corners[-self.nCols, 0]
        return (up_left, up_right, down_right, down_left)
    
    def getCorners(self, img):
        """Get corners from the image

        Args:
            img: input image to get corners
            isThermal (bool): image is thermal image (True) or not (False)
        Returns:
            retval: result (True if corners are extracted successfully, False otherwise)
            corners: corner list
        """        
        if self.img_shape == None:
            self.img_shape = img.shape[:2][::-1]
        else:
            assert self.img_shape == img.shape[:2][::-1], "All images must share the same size."
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nCols, self.nRows), flags=RGB_CORNER_FLAGS)
        
        # ret, corners = cv2.findChessboardCornersSB(gray, (self.nCols, self.nRows), flags=RGB_CORNER_FLAGS)
        
        if not ret:
            return ret, corners
        
        # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
        # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
        # of the image size. See http://answers.ros.org/question/3155/how-can-i-calibrate-low-resolution-cameras
        if not self.is_low_res and not all([(self.nBorders < corners[i, 0, 0] < (self.img_shape[1] - self.nBorders)) and (self.nBorders < corners[i, 0, 1] < (self.img_shape[0] - self.nBorders)) for i in range(corners.shape[0])]):
            ret = False
            
        # Ensure that all corner-arrays are going from top to bottom.
        if self.nCols != self.nRows:
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                corners = np.copy(np.flipud(corners))
        else:
            direction_corners = (corners[-1] - corners[0]) >= np.array([[0.0, 0.0]])
            if not np.all(direction_corners):
                if not np.any(direction_corners):
                    corners = np.copy(np.flipud(corners))
                elif direction_corners[0][0]:
                    corners = np.rot90(corners.reshape(self.nRows, self.nCols, 2)).reshape(self.nCols * self.nRows, 1, 2)
                else:
                    corners = np.rot90(corners.reshape(self.nRows, self.nCols, 2), 3).reshape(self.nCols * self.nRows, 1, 2)
        
        if ret:
            # Use a radius of half the minimum distance between corners. This should be large enough to snap to the
            # correct corner, but not so large as to include a wrong corner in the search window.
            min_distance = float("inf")
            for row in range(self.nRows):
                for col in range(self.nCols - 1):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + 1, 0]))
            for row in range(self.nRows - 1):
                for col in range(self.nCols):
                    index = row * self.nRows + col
                    min_distance = min(min_distance, _pdist(corners[index, 0], corners[index + self.nCols, 0]))
            radius = int(ceil(min_distance * 0.5))
            corners = cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1), SUBPIX_CRITERIA)
            
        return ret, corners
