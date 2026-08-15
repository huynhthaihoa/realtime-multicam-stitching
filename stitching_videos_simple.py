import cv2
import argparse
import time
from simple.stitcher import Stitcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=str, default='0,1,2', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--original', action='store_true', help="Show original video (True - default, False)")
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
    
    opt = parser.parse_args()
    print(opt)
    sources, output_path, show = opt.sources, opt.output_path, opt.original
    # if save and output_path == '':
    #     output_path = str(timestamp) + ".mp4"
    
    stitcher = Stitcher(opt.feature, opt.matching, opt.retention_thres, opt.reprojection_thres, opt.minimum_match_thres)
    
    if ',' in sources:
        sources = sources.split(',')
    
    caps = list()
    for source in sources:
        caps.append(cv2.VideoCapture(source))
    
    # Set Dataloader
    videoWriter = None
    
    while True:
        imgs = list()
        ret = True
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret is False:
                break
            if show:
                cv2.namedWindow(f"Camera {i}", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow(f"Camera {i}", frame.copy())
            imgs.append(frame)#.copy())
        if ret is False:
            break
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):  # q to quit
            break
        second = time.time()
        retval, stitched = stitcher.stitch(imgs)
        print("Stitching time: {} sec. Result: {}".format(time.time() - second, retval))
        if retval:
            if output_path != '':
                if videoWriter is None:# and retval:
                    w, h = stitched.shape[1], stitched.shape[0]
                    videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                videoWriter.write(stitched)   
            cv2.namedWindow("Stitched", flags = cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Stitched", stitched)
        else:
            continue
    if videoWriter is not None:
        videoWriter.release()
    else:
        cv2.destroyAllWindows()
    for cap in caps:
        cap.release()
    print("Finish!")
        
    
    


 

