import argparse
import time
import asyncio
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detect.utils.datasets import letterbox
from advanced.datasets import StreamReader, Camera
from advanced.stitcher import Stitcher
from detect.utils.general import (
    non_max_suppression, scale_coords, strip_optimizer)
from detect.utils.plots import plot_one_box
from detect.utils.torch_utils import select_device, time_synchronized

from detect.models import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def get_tensor_image(stitched, img_size, device, half):
    # Padded resize
    img = letterbox(stitched, new_shape=img_size, )[0]
    print("image shape:", img.shape)
        
    # Convert
    img = img.transpose(2, 0, 1) #convert 416x416x3 to 3x416x416
    #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
        
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

async def detect():
    sources, output_path, show, weights, imgsz, cfg, names,  = \
        opt.sources, opt.output_path, opt.original, opt.weights, opt.img_size, opt.cfg, opt.names,  \
    
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

    # Set up DeltaX logo
    deltax_img = cv2.imread('img/deltax.png')
    pts = list()
    h, w, _ = deltax_img.shape
    for i in range(h):
        for j in range(w):
            if deltax_img[i][j][0] != 0 or deltax_img[i][j][1] != 0 or deltax_img[i][j][2] != 0:
                pts.append((i, j))
    
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    except:
        load_darknet_weights(model, weights)
    model.to(device).eval()
    if half:
        model.half()  # to FP16
    
    cudnn.benchmark = True  # set True to speed up constant image size inference
    cudnn.enabled = True
    
    # Get names and colors
    names = load_classes(names)
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    videoWriter = None

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
        retval, stitched = stitcher.stitch(frames)
        if not retval:
            continue    
           
        img = get_tensor_image(stitched, imgsz, device, half) 
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = stitched
            s = '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '{}: {:.2f}%'.format(names[int(cls)], conf * 100)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Stream results
        for pt in pts:
            im0[pt[0], pt[1]] = deltax_img[pt[0], pt[1]]

        cv2.imshow("Stitched", im0)

        # Save video
        if output_path != '':
            if videoWriter is None:
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            videoWriter.write(im0)
    
    cv2.destroyAllWindows()        

    if videoWriter is not None:
        videoWriter.release()
    
    for source in sources:
        await connection.unregister_camera(source)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='detect/yolov7/yolov7-tiny.weights', help='model.pt path(s)')
    parser.add_argument('--sources', type=str, default='0,1,2', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--drop_rate', type=int, default=3, help='Drop rate. Default is 3.')
    parser.add_argument('--original', action='store_true', help="Show original video (True, False - default)")
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--save', action='store_true', help="Save output (True, False - default)")
    parser.add_argument('--output_path', type=str, default='result.mp4', help='Output video path (.mp4). Default is result.mp4')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='detect/yolov7/yolov7-tiny.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='detect/yolov7/coco.names', help='*.cfg path')
    parser.add_argument("--feature", help="Feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB - default, 4: SIFT, 5: SURF)", 
                        type=int, default=3)
    parser.add_argument("--thres", help="Feature retention threshold. The default value is 0.3 for ORB and 0.65 for other feature types.",
                    type=float, default=-1)
    # parser.add_argument("--roi", help="Cropped ROI", nargs="+", type=int)
    parser.add_argument("--seam_choice", help="Seam estimation type (0: dp_color, 1: dp_colorgrad, 2: voronoi, 3: no). The default is 2 (voronoi).",
                    type=int, default=2)
    parser.add_argument("--wave_correct_choice", help="Wave effect correction type (0: horizontal, 1: none, 2: vertical). The default is 1 (none).",
                    type=int, default=1)

    opt = parser.parse_args()
    print(opt)
    

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                asyncio.run(detect())
                strip_optimizer(opt.weights)
        else:
            asyncio.run(detect())#opt.feature_extractor_method, opt.matching_method, opt.feature_retention_thres, opt.reprojection_thres)
