import asyncio
import cv2
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# from advanced.datasets import LoadStreams
from advanced.stream import StreamReader, Camera
from advanced.stitcher import Stitcher

from tracker import *

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

async def depth():
    sources, output_path, show, imgsz, model_type,  device_choice = \
        opt.sources, opt.output_path, opt.original, opt.img_size, opt.model_type, opt.device
    featureExtractor, matchThres = opt.feature, opt.thres
    seamChoice, waveCorrectChoice = opt.seam_choice, opt.wave_correct_choice
    
    stitcher = Stitcher(featureExtractor, matchThres, seam_choice=seamChoice, wave_correct_choice=waveCorrectChoice)

    if ',' in sources:
        sources = sources.split(',')
    
    # Set up camera list
    connection = StreamReader()    
    camera_conn = [connection.register_camera(Camera(), source) for source in sources]
    await asyncio.gather(*camera_conn)
    await connection.set_recorder(sources)
    
    # Set up videoWriter
    videoWriter = None
                
    # w = -1
    # h = -1
    # x = -1
    # y = -1
    
    # if opt.roi is not None and len(opt.roi) == 4:
    #     x, y, w, h = opt.roi
        
    # Initialize
    device = torch.device("cuda" if (torch.cuda.is_available() and device_choice != '') else "cpu")

    # Load depth estimation model
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
        
    # Load detection model
    detection = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # detection.to(device)
    # detection.eval()
    
    # Set Dataloader
    videoWriter = None
    cudnn.benchmark = True  # set True to speed up constant image size inference
    cudnn.enabled = True
    #dataset = LoadStreams(drop_rate, featureExtractor, matchingMethod, retentionThres, reprojThresh, source, x, y, w, h, img_size=imgsz)
    # dataset = LoadStreams(3, -1, source, x, y, w, h)#, img_size=imgsz)
    
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=30,
    )
    
    # Run inference
    t0 = time.time()     
    new_frame_time = t0
    while True:
        
        prev_frame_time = new_frame_time
        new_frame_time = time.time()
        # Calculating the fps
        if new_frame_time != prev_frame_time:
            fps = 1 / (new_frame_time - prev_frame_time)
            print("FPS: ", fps)
            
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):  # q to quit
            break   
        msgs = await connection.read_streams(sources)
        frames = list() # list of current frames
        for msg in msgs:
            if msg[1] is None:
                break
            else:
                frames.append(msg[1].copy())
                if show:
                    cv2.imshow(msg[0], msg[1])    
        retval, img = stitcher.stitch(frames)
        if not retval:
            continue
    # # Run inference
    # t0 = time.time()
    # for retval, img, img0s in dataset:
        #img_input = cv2.resize(img, (384, 384))
        img_input = transform(img).to(device)# transform({"image": img})["image"]
        #img_input = img.to(device)

        # Depth estimation
        t1 = time_synchronized()
        
        depthPred = model(img_input)
        depthPred = torch.nn.functional.interpolate(
                    depthPred.unsqueeze(1),
                    size=(img.shape[0], img.shape[1]),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
        
        depth_map = depthPred.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_v = depth_map 
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        t2 = time_synchronized()
        
        # Print time (inference + NMS)
        print('Depth Estimation Done. (%.3fs)' % (t2 - t1))
    
        # Object detection
        detPred = detection(img)
        
        detections = yolo_detections_to_norfair_detections(detPred, track_points='bbox')
        tracked_objects = tracker.update(detections=detections)
        
        frame = img
        
        norfair.draw_points(frame, detections)
        norfair.draw_boxes(frame, detections, line_color=(10, 246, 255), line_width=1)
        norfair.draw_tracked_objects(frame, tracked_objects, color=(0, 0, 255), id_size=.51)  # 250, 246, 10

        for box in detPred.xyxy[0]:
            if box[5] == 2:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                xCenter = xA + (xB - xA) // 2
                yCenter = yA + (yB - yA) // 2

                print(xCenter, yCenter)

                depth_value = ((1 - depth_v[yCenter][xCenter]) * 2 ** (
                                    1 - depth_v[yCenter][xCenter])) * 12  # using the joutron camera

                (h, w, c) = img.shape

                cv2.putText(img, ' {: .0f} m'.format(depth_value), (xA, yA),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (2, 242, 122), 1, cv2.LINE_AA)  # 122, 242, 2
                cv2.line(img, (w // 2, h), (xCenter, yCenter), (242, 82, 2), 1)
                
        view = np.vstack((img, depth_map))
                
        cv2.imshow("Stitched", view)
        # if show:
        #     for j, img0 in enumerate(img0s):
        #         cv2.imshow("Camera {}".format(j), img0)
        
        # Save video
        if output_path != '':
            if videoWriter is None:
                fps, w, h = 30, view.shape[1], view.shape[0]
                videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            videoWriter.write(view)
                    
        # if cv2.waitKey(1) == ord('q'):  # q to quit
        #     raise StopIteration
    cv2.destroyAllWindows()        
    if videoWriter is not None:
        videoWriter.release()
    
    for source in sources:
        await connection.unregister_camera(source)

    print('Done. (%.3fs)' % (time.time() - t0))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', default='MiDaS_small', help='depth estimation model type: DPT_Large, DPT_Hybrid or MiDaS_small')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sources', type=str, default='0,1,2', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--drop_rate', type=int, default=3, help='Drop rate. Default is 3.')
    parser.add_argument('--original', action='store_true', help="Show original video (True, False - default)")
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--save', action='store_true', help="Save output (True, False - default)")
    parser.add_argument('--output_path', type=str, default='result.mp4', help='Output video path (.mp4). Default is result.mp4')
    parser.add_argument("--feature", help="Feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB - default, 4: SIFT, 5: SURF)", 
                        type=int, default=3)
    # parser.add_argument("--matching_method", help="Matching method (0: least-square method, 1: RANSAC method - default, 2: Least-Median robust method, 3: PROSAC-based robust method)", 
    #                     type=int, default=1)
    parser.add_argument("--thres", help="Feature retention threshold. The default value is 0.3 for ORB and 0.65 for other feature types.",
                    type=float, default=-1)
    # parser.add_argument("--reprojection_thres", help="Reprojection threshold. Default is 4",
    #                 type=int, default=4)
    parser.add_argument("--seam_choice", help="Seam estimation type (0: dp_color, 1: dp_colorgrad, 2: voronoi, 3: no). The default is 2 (voronoi).",
                    type=int, default=2)
    parser.add_argument("--wave_correct_choice", help="Wave effect correction type (0: horizontal, 1: none, 2: vertical). The default is 1 (none).",
                    type=int, default=1)
    # parser.add_argument("--roi", help="Cropped ROI", nargs="+", type=int)

    opt = parser.parse_args()
    print(opt)
    
    with torch.no_grad():
        asyncio.run(depth())