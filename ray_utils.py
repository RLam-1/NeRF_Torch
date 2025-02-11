# This file contains the APIs for calculating ray directions
# Ray directions are used to train the neural radiance field by capturing
# the RGB and volume density at that particular 3d point

import numpy as np
import torch
import torch.nn as nn

def gen_img_rays(H, W, f, cam_2_world):
  # assume the camera frame of reference is right, up, out
  # this means that the as we move along H the y-val is negative
  # assume z is 1
  # NOTE - the pixels are in image space - need to convert to camera space by reversing
  # camera intrinsics operations 
  # cx = W * 0.5, cy = H * 0.5
  
  #  x_camera = (x_image - cx)/f, y_camera = (y_image - cy)/f
  i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
  cx, cy = W * 0.5, H * 0.5
  
  # The ray dirs are in row order [(0,0,1) -> (W,0,1), (0,1,1) -> (W,1,1)]
  raydirs = np.stack([(i-cx)/f, -(j-cy)/f, -np.ones_like(i)], -1)
  pixelDict = np.stack([i, j], -1)
  
  # multiply the cam_2_world rotation matrix to get the direction of the ray
  # in world space
  # NOTE: the top left 3x3 matrix is the rotation matrix
  ray_dirs_w = raydirs @ cam_2_world[:3, :3].T
  
  # The origin of the ray is origin of the camera which is just the translation vector
  # (the top right [3,1] of the cam_2_world matrix)
  ray_origs_w = np.broadcast_to(cam_2_world[:3, -1], np.shape(ray_dirs_w))
  
  # reshape the ray dirs and ray origins in world space to collapse the 1st 2 dimensions
  # so that it is (num of pixels W * H by 3 <- vector) or 2 <- pixelDict 
  return np.reshape(ray_dirs_w, [-1,3]), np.reshape(ray_origs_w, [-1,3]), np.reshape(pixelDict, [-1,2])

# This generates the points along the ray 
def gen_ray_pts(ray_orig, ray_dir, near, far, N_points):
  t_vals = np.linspace(0., 1., N_points)
  # let's do sample linearly in inverse depth (disparity)
  z_vals = near * (1. - t_vals) + far * (t_vals)
  pts = ray_orig + ray_dir * z_vals[..., :, None]
  # return the zvals in their original 1 by N dimension because the zvals are just from 1 sample ray and each entry has only 1 dimension
  return pts, z_vals

