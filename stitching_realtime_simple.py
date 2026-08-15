import cv2
import argparse
import time
from simple.datasets import LoadStreams
#import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=str, default='0,1,2', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--original', action='store_false', help="Show original video (True - default, False)")
    #parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--save', action='store_true', help="Save output (True, False - default)")
    parser.add_argument('--output_path', type=str, default='', help='Output video path (.mp4). Default is result.mp4')
    parser.add_argument("--feature", help="Feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB, 4: SIFT - default)", 
                        type=int, default=4)
    parser.add_argument("--matching", help="Matching method (0: least-square method, 1: RANSAC method - default, 2: Least-Median robust method, 3: PROSAC-based robust method)", 
                        type=int, default=1)
    parser.add_argument("--retention_thres", help="Feature matching retention threshold. The default value is 0.3 for ORB and 0.65 for other feature types",
                    type=float, default=-1)
    parser.add_argument("--reprojection_thres", help="Reprojection threshold. Default is 4",
                    type=int, default=4)
    parser.add_argument("--minimum_match_thres", help="Minimum number of matched points to be considered as a good match. Default is 10",
                    type=int, default=10)
    #parser.add_argument("--roi", help="Cropped ROI", nargs="+", type=int)
    
    opt = parser.parse_args()
    print(opt)
    sources, output_path, show = opt.sources, opt.output_path, opt.original
    featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres = opt.feature, opt.matching, opt.retention_thres, opt.reprojection_thres, opt.minimum_match_thres
    #roi = opt.roi
    
    # if save and output_path == '':
    #     output_path = str(time.time()) + ".mp4"
    # x = -1
    # y = -1
    # w = -1
    # h = -1
    
    # if roi is not None and len(roi) == 4:
    #     x, y, w, h = roi
    
    if ',' in sources:
        sources = sources.split(',')
    # n = len(sources)
    # videoWriters = [None] * n
    
    # output_raw = str(time.time())
    # os.mkdir(output_raw)
    
    # Set Dataloader
    videoWriter = None
    dataset = LoadStreams(sources, featureExtractor, matchingMethod, retentionThres, reprojThresh, matchThres)#, x, y, w, h)
    #fps = dataset.fps
    print('Finish initialization!')
    #cnt = 0
    new_frame_time = time.time()
    for err, img, im0s in dataset:
        prev_frame_time = new_frame_time
        new_frame_time = time.time()
        # Calculating the fps
        if new_frame_time != prev_frame_time:
            fps = 1 / (new_frame_time - prev_frame_time)
            print("FPS: ", fps)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):  # q to quit
            break
        if err == -1:
            continue
        if not err:
            cv2.namedWindow("Stitched", flags = cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Stitched", img)
            # Save video
            if output_path != '': 
                if videoWriter is None:
                    w, h = img.shape[1], img.shape[0]
                    videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                videoWriter.write(img)
        if show:
            for j, vid in enumerate(im0s):
                cv2.namedWindow(f"Camera {j}", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(f"Camera {j}", vid)
                # if videoWriters[j] is None:
                #     fps, w, h = 30, vid.shape[1], vid.shape[0]
                #     videoWriters[j] = cv2.VideoWriter("{}/video_{}.mp4".format(output_raw, j), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                # videoWriters[j].write(vid)

        #time.sleep(1 / fps)
    cv2.destroyAllWindows()        
        
        #cnt += 1
    if videoWriter is not None:
        videoWriter.release()
        
    # for video in videoWriters:
    #     video.release()
        
    
    


 

