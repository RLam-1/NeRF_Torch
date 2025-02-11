# This class loads a blender file which has JSON format and contains
# the following fields:
#
#   "camera_angle_x" -> int
#   "frames" -> list of JSON
#
#   Each entry in "frames" contains:
#      "file_path" -> relative path of the image
#      "rotation"
#      "transform_matrix" -> camera to world matrix
#
import os
import json
import numpy as np
import torch
import torch.nn as nn
import imageio
from ray_utils import *

class imgObject:
  def __init__(self, image, cam_2_world, height, width, fx, fy, near=0, far=1):
    self.image = image
    self.cam_2_world = cam_2_world
    self.height = height
    self.width = width
    self.fx = fx
    self.fy = fy
    self.near = near
    self.far = far
    
  # the model data from image objects is in format
  #  [ray_orig<x,y,z>, ray_dir <x,y,z>, near_far<near, far>, pixel_color <r,g,b,a>] - total of 12 entries
  def genModelData(self):
    ray_dirs, ray_origs, _ = gen_img_rays(self.height, self.width, self.fx, self.cam_2_world)
    imgs = self.image.reshape(-1,4)
    near_far = np.array([self.near, self.far])
    near_far = np.broadcast_to(near_far, (ray_dirs.shape[0], near_far.shape[0]))
    model_data = np.concatenate([ray_origs, ray_dirs, near_far, imgs], axis=-1)
    return model_data

class LoadBlender:

  def __init__(self, jsonFile, imgSuffix=".png"):
    self.jsonFile = jsonFile
    self.rootDir = os.path.dirname(jsonFile)
    self.imgSuffix = imgSuffix
    self.imgObjects = []
    
  def rtrvImgObjects(self):
    with open(self.jsonFile) as jsFile:
      blenderJSON = json.load(jsFile)
      for frame in blenderJSON["frames"]:
        # read the image from the file name
        imgName = os.path.join(self.rootDir, frame["file_path"] + self.imgSuffix)
        img = imageio.imread(imgName)
        # normalize the image value and keep all 4 channels (RGBA)
        img = (np.array(img) / 255.).astype(np.float32)
        # get the cam_2_world matrix
        cam_2_world = np.array(frame["transform_matrix"]).astype(np.float32)
        # calculate height, width of the image
        H, W = img.shape[:2]
        # calculate the focal length of the camera, which is
        #  tan(camera_angle_x * 0.5) = W * 0.5 / f
        fx = fy = W * 0.5 / np.tan(float(blenderJSON["camera_angle_x"]) * 0.5)
        self.imgObjects.append(imgObject(img, cam_2_world, H, W, fx, fy))
        
    return self.imgObjects

