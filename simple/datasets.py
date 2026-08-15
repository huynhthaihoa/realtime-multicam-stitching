# Dataloaders

import os
import time
from threading import Thread

import cv2
import numpy as np

from simple.stitcher import Stitcher

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources, featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres):#, x=-1, y=-1, w=-1, h=-1):#, img_size=640):
        # self.x = x
        # self.y = y
        # self.w = w
        # self.h = h
        self.stitcher = Stitcher(featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres)
        if isinstance(sources, list) is False:
            if os.path.isfile(sources):
                with open(sources, 'r') as f:
                    sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
            else:
                sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        #self.locks = [True] * n
        self.sources = sources
        #self.imgs = [[None] * n for i in range(1800)]# np.zeros((30, n))
        self.cacheIdx = 0
        self.fps = None
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            ret = cap.isOpened()
            assert ret, 'Failed to open %s' % s
            if not ret:
                break 
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            if self.fps is None:
                self.fps = fps
            _, img = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

    def update(self, index, cap):
        '''
        Read next stream frame in a daemon thread:
        '''
        #n = 0
        while cap.isOpened():
            #n += 1
            # _, self.imgs[index] = cap.read()
            ret = cap.grab()
            if ret:
                success, im = cap.retrieve()
                self.imgs[index] = im if success else None
            else:
                raise StopIteration
            # if ret and not self.locks[index]:#n == self.frequency:  # read every 4th frame
            #     success, im = cap.retrieve()
            #     # if success:
            #     #     self.imgs[index] = im
            #     # else:
            #     #     raise StopIteration
            #     self.imgs[index] = im if success else None
            #     self.locks[index] = True
                #n = 0
                
            #time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0s = self.imgs.copy()
        # for idx, img in enumerate(img0s):
        #     if img is None:
        #         raise StopIteration
            #self.locks[idx] = False
        #self.cacheIdx = (self.cacheIdx + 1) % 1800
        # if self.cacheIdx == 1800:
        #     self.cacheIdx = 0
        for img in img0s:
            if img is None:
                return -1, None, None

        second = time.time()
        #Do stitching here
        try:
            #retval, stitched = self.stitcher.stitch(img0s)
            success, stitched = self.stitcher.stitch(img0s)
            ret = 0
            if not success:
                #stitched = img0s[1]
                ret = 1
            # else:#if self.x != -1: # crop ROI
            #     stitched = stitched[self.y : self.y + self.h, self.x : self.x + self.w]
        except:
            #stitched = img0s[1]
            ret = 2
        
        # succeed, stitched = self.stitcher.stitch(img0s)
        # if not succeed:
        #     stitched = img0s[1]
        #     retval = False
        # elif self.x != -1:
        #     stitched = stitched[self.y : self.y + self.h, self.x : self.x + self.w]
        print("Stitching time: {} sec".format(time.time() - second))
        return ret, stitched, img0s

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years