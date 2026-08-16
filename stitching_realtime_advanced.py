import asyncio
import cv2
import argparse
import time
from advanced.datasets import StreamReader, Camera
from advanced.stitcher import Stitcher

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=str, default='0,1,2', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--original', action='store_true', help="Show original video (True - default, False)")
    parser.add_argument('--save', action='store_true', help="Save output (True, False - default)")
    parser.add_argument('--output_path', type=str, default='', help='Output video path (.mp4). Default is result.mp4')
    parser.add_argument("--feature", help="Feature extractor method (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB - default, 4: SIFT)", 
                        type=int, default=3)
    parser.add_argument("--thres", help="Feature matching retention threshold. The default value is 0.3 for ORB and 0.65 for other feature types",
                    type=float, default=-1)
    parser.add_argument("--seam_choice", help="Seam estimation type (0: dp_color, 1: dp_colorgrad, 2: voronoi, 3: no). The default is 2 (voronoi).",
                    type=int, default=2)
    parser.add_argument("--wave_correct_choice", help="Wave effect correction type (0: horizontal, 1: none, 2: vertical). The default is 1 (none).",
                    type=int, default=1)
    
    opt = parser.parse_args()
    print(opt)
    # timestamp = str(time.time())
    # os.mkdir("detailed/" + timestamp)
    sources, save, output_path, show = opt.sources, opt.save, opt.output_path, opt.original
    # if save and output_path == '':
    #     output_path = str(timestamp) + ".mp4"
    featureExtractor, matchThres = opt.feature, opt.thres
    seamChoice, waveCorrectChoice = opt.seam_choice, opt.wave_correct_choice
    
    stitcher = Stitcher(featureExtractor, matchThres, seam_choice=seamChoice, wave_correct_choice=waveCorrectChoice)
     
    if ',' in sources:
        sources = sources.split(',')
    # sources = [sources]
        
    # Set up camera list
    connection = StreamReader()    
    camera_conn = [connection.register_camera(Camera(), source) for source in sources]
    await asyncio.gather(*camera_conn)
    await connection.set_recorder(sources)
    
    # Set up videoWriter
    videoWriter = None
                
    print('Finish initialization!')
    
    new_frame_time = time.time()
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
                    cv2.namedWindow(msg[0], flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.imshow(msg[0], msg[1])
        retval, stitched = stitcher.stitch(frames)
        if retval:
            if save:
                if videoWriter is None:# and retval:
                    w, h = stitched.shape[1], stitched.shape[0]
                    videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                videoWriter.write(stitched)
            cv2.namedWindow("Stitched", flags = cv2.WINDOW_FULLSCREEN)           
            cv2.imshow("Stitching result", stitched)
    
    cv2.destroyAllWindows()        
        
    if videoWriter is not None:
        videoWriter.release()
    
    for source in sources:
        await connection.unregister_camera(source)


if __name__ == "__main__":
    asyncio.run(main())        
    
    


 

