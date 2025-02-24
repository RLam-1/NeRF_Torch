# This file contains the APIs for calculating ray directions
# Ray directions are used to train the neural radiance field by capturing
# the RGB and volume density at that particular 3d point

import numpy as np
import torch
import torch.nn as nn

def gen_img_rays(H, W, f, cam_2_world, crop_factor=0.5):
  # assume the camera frame of reference is right, up, out
  # this means that the as we move along H the y-val is negative
  # assume z is 1
  # NOTE - the pixels are in image space - need to convert to camera space by reversing
  # camera intrinsics operations 
  # cx = W * 0.5, cy = H * 0.5
  
  #  x_camera = (x_image - cx)/f, y_camera = (y_image - cy)/f
  i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
  cx, cy = W * 0.5, H * 0.5
  
  dW, dH = int(W//2 * crop_factor), int(H//2 * crop_factor)
  cropped_i, cropped_j = torch.meshgrid(torch.arange(W//2 - dW, W//2 + dW), torch.arange(H//2 - dH, H//2 + dH), indexing='xy')

  # The ray dirs are in row order [(0,0,1) -> (W,0,1), (0,1,1) -> (W,1,1)]
  raydirs = torch.stack(((i-cx)/f, -(j-cy)/f, -torch.ones_like(i)), -1)
  cropped_raydirs = torch.stack(((cropped_i-cx)/f, -(cropped_j-cy)/f, -torch.ones_like(cropped_i)), -1)
  print(raydirs.is_cuda)
  pixelDict = torch.stack((i, j), -1)
  
  # multiply the cam_2_world rotation matrix to get the direction of the ray
  # in world space
  # NOTE: the top left 3x3 matrix is the rotation matrix
  ray_dirs_w = torch.sum(raydirs[..., np.newaxis, :] * cam_2_world[:3,:3], dim=-1)
  cropped_ray_dirs_w = torch.sum(cropped_raydirs[..., np.newaxis, :] * cam_2_world[:3,:3], dim=-1)
  print("ray_dirs_w {}".format(ray_dirs_w))
  ray_dirs_w2 = raydirs @ cam_2_world[:3, :3].T
  print("ray_dirs_w2 {}".format(ray_dirs_w2))
  # The origin of the ray is origin of the camera which is just the translation vector
  # (the top right [3,1] of the cam_2_world matrix)
  ray_origs_w = torch.broadcast_to(cam_2_world[:3, -1], ray_dirs_w.shape)
  cropped_ray_origs_w = torch.broadcast_to(cam_2_world[:3, -1], cropped_ray_dirs_w.shape)
  # reshape the ray dirs and ray origins in world space to collapse the 1st 2 dimensions
  # so that it is (num of pixels W * H by 3 <- vector) or 2 <- pixelDict 
  return torch.reshape(ray_dirs_w, [-1,3]), \
         torch.reshape(ray_origs_w, [-1,3]), \
         torch.reshape(cropped_ray_dirs_w, [-1,3]), \
         torch.reshape(cropped_ray_origs_w, [-1,3])

# This generates the points along the ray 
def gen_ray_pts(ray_orig, ray_dir, near, far, N_points):
  ray_orig = torch.from_numpy(ray_orig)
  ray_orig = ray_orig.to('cuda')
  ray_dir = torch.from_numpy(ray_dir)
  ray_dir = ray_dir.to('cuda')
  t_vals = np.linspace(0., 1., N_points)
  # let's do sample linearly in inverse depth (disparity)
  z_vals = near * (1. - t_vals) + far * (t_vals)
  z_vals_tensor = torch.from_numpy(z_vals[..., :, None])
  z_vals_tensor = z_vals_tensor.to('cuda')
  pts = ray_orig + ray_dir * z_vals_tensor
  # return the zvals in their original 1 by N dimension because the zvals are just from 1 sample ray and each entry has only 1 dimension
  return pts, z_vals

# This generates the points along the ray 
def gen_pts_from_rays_batch(ray_orig, ray_dir, near, far, N_points):
  # need to convert ray_orig, ray_dir into 3D arrays
  # of shape (ray_orig.shape[0], 1, ray_orig.shape[-1]) -> add extra dimension in the middle of dimension 1
  #ray_orig = torch.from_numpy(ray_orig.reshape(ray_orig.shape[0], 1, ray_orig.shape[-1]))
  #ray_orig = ray_orig.to('cuda')
  ray_orig = ray_orig.reshape(ray_orig.shape[0], 1, ray_orig.shape[-1])
  #ray_dir = torch.from_numpy(ray_dir.reshape(ray_dir.shape[0], 1, ray_dir.shape[-1]))
  ray_dir = ray_dir.reshape(ray_dir.shape[0], 1, ray_dir.shape[-1])
  #ray_dir = ray_dir.to('cuda')
  t_vals = torch.linspace(0., 1., N_points)
  # let's do sample linearly in inverse depth (disparity)
  z_vals = near * (1. - t_vals) + far * (t_vals)
  z_vals_t = z_vals[..., :, None]
  z_vals_t += torch.rand(z_vals_t.shape) * (far[0][0]-near[0][0])/N_points
  print("z_vals_t shape is {}".format(z_vals_t.shape))
 # z_vals_tensor = torch.from_numpy(z_vals[..., :, None])
 # z_vals_tensor = z_vals_tensor.to('cuda')
  pts = ray_orig + ray_dir * z_vals_t
  # return the zvals in their original 1 by N dimension because the zvals are just from 1 sample ray and each entry has only 1 dimension
  return pts, z_vals