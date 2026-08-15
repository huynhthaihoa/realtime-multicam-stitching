# Dataset utils and dataloaders

import os
from threading import Thread

import cv2
import numpy as np
import torch

from advanced.stitcher import Stitcher

class LoadStreams:  
    def __init__(self, sources, featureExtractor, retentionThres, device, half, x=-1, y=-1, w=-1, h=-1, img_size=640):# featureExtractor, matchingMethod, retentionThres, reprojThresh, sources, x=-1,y=-1,w=-1, h=-1):#, img_size=640):
        self.img_size = img_size
        self.stitcher = Stitcher(featureExtractor, matchThres=retentionThres)
        self.device = device
        self.half = half
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if isinstance(sources, list) is False:
            if os.path.isfile(sources):
                with open(sources, 'r') as f:
                    sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
            else:
                sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.cnts = [True] * n
        self.sources = sources
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
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        '''
        Read next stream frame in a daemon thread:
        '''
        #n = 0
        while cap.isOpened():
            #n += 1
            # _, self.imgs[index] = cap.read()
            ret = cap.grab()
            if ret and not self.cnts[index]:#n == self.frequency:  # read every 4th frame
                success, im = cap.retrieve()
                # if success:
                #     self.imgs[index] = im
                # else:
                #     raise StopIteration
                self.imgs[index] = im if success else None
                self.cnts[index] = True
                #n = 0
            #time.sleep(0.1)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        imgs = self.imgs.copy()
        for idx, img in enumerate(imgs):
            if img is None:
                raise StopIteration
            self.cnts[idx] = False
        # if cv2.waitKey(1) == ord('q'):  # q to quit
        #     cv2.destroyAllWindows()
        #     raise StopIteration

        #Do stitching here
        stitched = self.stitcher.stitch(imgs)
        retval = True
        if stitched is None:
            retval = False
            stitched = imgs[1].copy()
        elif self.x != -1:
            stitched = stitched[self.y : self.y + self.h, self.x : self.x + self.w]
        
        # Padded resize
        img = letterbox(stitched, new_shape=self.img_size)[0]
        print("image shape:", img.shape)
        
        # Convert
        img = img.transpose(2, 0, 1) #convert 416x416x3 to 3x416x416
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return retval, img, stitched, imgs

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
