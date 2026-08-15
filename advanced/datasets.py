# Dataloaders

import asyncio
import os

import cv2

from advanced.stitcher import Stitcher

from typing import Any, List, Protocol

class Stream(Protocol):
  async def connect(self, parent_dir, camera_id) -> None:
    ...

  async def disconnect(self) -> None:
    ...
  
  async def run_stream(self, timestamp: str) -> Any:
    ...

class Camera:
  """
  """  
  async def connect(self, camera_id) -> None:
    self.camera_id = camera_id    
    if camera_id.isnumeric():
      self.camera_name = f"camera {self.camera_id}"
      self.stream = cv2.VideoCapture(eval(camera_id))
    else:
      self.camera_name = f"video {self.camera_id}"
      self.stream = cv2.VideoCapture(camera_id)
    print("Connecting", self.camera_name)
    if not self.stream.isOpened():
      print("Error when opening", self.camera_name)
      return    
  
  async def disconnect(self) -> None:
    print("Disconnecting", self.camera_name)
    self.stream.release()
    self.record = False
  
  async def set_recorder(self) -> None:
    """Begin recording
    """    
    self.record = True 
    
  async def get_id(self):
    """Get camera id

    Returns:
        camera_id: camera id
    """    
    return self.camera_id 
  
  async def get_name(self):
    """Get camera name

    Returns:
        camera_name: camera name
    """    
    return self.camera_name 
  
  async def run_stream(self) -> Any:  

    ret, frame = self.stream.read()
    return self.camera_name, frame

class StreamReader:
  def __init__(self) -> None:
    """StreamReader __init__ 

    Args:
        root_folder (str): root folder to save data
    """    
    self.streams: dict[str, Stream] = {} 

  async def register_camera(self, camera: Stream, camera_id: str) -> str:
    """register camera

    Args:
        camera (cameraStream): camera
        camera_id (str):camera id

    Returns:
        str: camera_name
    """    
    await camera.connect(camera_id)
    
    camera_name = await camera.get_id()
    self.streams[camera_name] = camera
    
    return camera_name

  async def unregister_camera(self, camera_name: str) -> None:
    await self.streams[camera_name].disconnect()
    del self.streams[camera_name]

  async def read_streams(self, cameras_name: Any) -> Any:
    """Get current frames

    Args:
        cameras_name (Any): sources name
    """    
    return await asyncio.gather(*[self.streams[camera_name].run_stream() for camera_name in cameras_name])
  
  async def set_recorder(self, cameras_name: List[str]) -> None:
    # print("Setting the name of files for saving")
    return await asyncio.gather(*[self.streams[camera_name].set_recorder() for camera_name in cameras_name])


class LoadStreams: # multiple IP or RTSP cameras
    def __init__(self, sources, featureExtractor, matchThres, seamChoice, waveCorrectChoice) -> None:
        # self.stitcher = cv2.Stitcher_create()
        self.stitcher = Stitcher(featureExtractor, matchThres, seam_choice = seamChoice, wave_correct_choice = waveCorrectChoice)#, wave_correct_choice = 1, warp_surface_type='plane')
        if isinstance(sources, list) is False:
            if os.path.isfile(sources):
                with open(sources, 'r') as f:
                    self.sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
            else:
                self.sources = [sources]
        else:
          self.sources = sources
          
        self.n_sources = len(self.sources)
    
    async def connect(self):
        self.connection = StreamReader()
        cameras = [Camera()] * self.n_sources
        self.all_cameras = [self.connection.register_camera(cameras[i], self.sources[i]) for i in range(self.n_sources)]
        asyncio.gather(*self.all_cameras)
        await self.connection.set_recorder(self.sources)  
        
    def __iter__(self):
      return self
      
    async def __next__(self):
        """retrieve current frames & stitching result

        Returns:
            ret: stitching result
            images: current frames
            sources: camera ids
            stitched: stitching result
        """        
        msgs = await self.connection.read_streams(self.sources)
        images = list()
        sources = list()
        for msg in msgs:
            if msg[1] is None:
                return -1, None, None, None
            else:
                sources.append(msg[0])
                images.append(msg[1])
        try:
            success, stitched = self.stitcher.stitch(images)
            ret = 0
            if not success:
                stitched = images[0]
                ret = 1
        except:
            stitched = images[1]
            ret = 2     
        return ret, images, sources, stitched
        
    async def disconnect(self):
        for source in self.sources:
            await self.connection.unregister_camera(source)
