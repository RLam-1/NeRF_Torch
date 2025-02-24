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
from PIL import Image

class imgObject:
  def __init__(self, image, cam_2_world, height, width, fx, fy, near, far, cropped_factor):
    self.image = image
    self.cam_2_world = cam_2_world
    self.height = height
    self.width = width
    self.fx = fx
    self.fy = fy
    self.near = near
    self.far = far
    self.cropped_factor = cropped_factor

  # return the rgb image - crop out the 'a' field
  def getImgRGB(self):
    return self.image[:,:,:3]

  def genRaysData(self):
    ray_dirs, ray_origs, cropped_ray_dirs, cropped_ray_origs = gen_img_rays(self.height, self.width, self.fx, self.cam_2_world, self.cropped_factor)
    near_far = torch.asarray([self.near, self.far])
    cropped_near_far = torch.broadcast_to(near_far, (cropped_ray_dirs.shape[0], near_far.shape[0]))
    near_far = torch.broadcast_to(near_far, (ray_dirs.shape[0], near_far.shape[0]))
    ray_data = torch.concatenate([ray_origs, ray_dirs, near_far], dim=-1)
    cropped_ray_data = torch.concatenate([cropped_ray_origs, cropped_ray_dirs, cropped_near_far], dim=-1)
    return ray_data, cropped_ray_data
    
  # the model data from image objects is in format
  #  [ray_orig<x,y,z>, ray_dir <x,y,z>, near_far<near, far>, pixel_color <r,g,b,a>] - total of 12 entries
  def genModelData(self):
    ray_data, cropped_ray_data = self.genRaysData()
    dH = int(self.height // 2 * self.cropped_factor)
    dW = int(self.width // 2 * self.cropped_factor)
   # self.image = self.image[...,:3]*self.image[..., -1:] + (1.-self.image[..., -1:])
    print(self.image.shape)
    cropped_img = self.image[self.height//2-dH : self.height//2+dH, self.width//2-dW : self.width//2+dW, :].reshape(-1,self.image.shape[-1])[...,:3]
    imgs = self.image.reshape(-1,self.image.shape[-1])[...,:3] # the imgs are r,g,b,a <- leave out the a component
    model_data = torch.concatenate([ray_data, imgs], dim=-1)
    cropped_model_data = torch.concatenate([cropped_ray_data, cropped_img], dim=-1)
    return model_data, cropped_model_data

class LoadBlender:

  def __init__(self, jsonFile, imgScale, near, far, cropped_factor=0.5, imgSuffix=".png"):
    self.jsonFile = jsonFile
    self.rootDir = os.path.dirname(jsonFile)
    self.imgSuffix = imgSuffix
    self.imgScale = imgScale
    self.near = near
    self.far = far
    self.cropped_factor = cropped_factor
    self.imgObjects = []

  def getNumOfImgs(self):
    return len(self.imgObjects)
    
  def rtrvImgObjects(self):
    with open(self.jsonFile) as jsFile:
      blenderJSON = json.load(jsFile)
      for frame in blenderJSON["frames"]:
        # read the image from the file name
        imgName = os.path.join(self.rootDir, frame["file_path"] + self.imgSuffix)
        #img = imageio.imread(imgName)
        with Image.open(imgName) as im:
          img = im.resize((int(im.width // self.imgScale), int(im.height // self.imgScale)), Image.Resampling.LANCZOS)
          # normalize the image value and keep all 4 channels (RGBA)
          img = np.asarray(img)
          img = torch.Tensor(img / 255.).to(torch.float32)
          print("img shape {}".format(img.shape))
        # get the cam_2_world matrix
        cam_2_world = torch.Tensor(frame["transform_matrix"]).to(torch.float32)
        # calculate height, width of the image
        H, W = img.shape[:2]
        # calculate the focal length of the camera, which is
        #  tan(camera_angle_x * 0.5) = W * 0.5 / f
        fx = fy = W * 0.5 / np.tan(float(blenderJSON["camera_angle_x"]) * 0.5)
        self.imgObjects.append(imgObject(img, cam_2_world, H, W, fx, fy, self.near, self.far, self.cropped_factor))
        
    return self.imgObjects

