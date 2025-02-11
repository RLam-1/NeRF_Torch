# This file contains the APIs for rendering

import numpy as np
import torch
import tensorflow as tf

# This API calculates the final rgb from the rgb_vol_density readings along a ray
def volumetric_render(rgb_vol, zvals, ray_dir):
  # first calculate the distances between the zvals of the points along the ray
  dists = zvals[..., 1:] - zvals[..., :-1]
  # need to add infinite for the distance of the last point
  dists = np.concatenate([dists, [1e10]], axis=-1)
  #dists = np.concatenate([dists, np.broadcast_to([1e10], (dists.shape[0]))], axis=-1)
  
  # multiply the distance by the norm of the ray dir to get the distance in world space
  dists *= np.linalg.norm(ray_dir, axis=-1)
 # print("rgb_vol is {}".format(rgb_vol))
#  print("dists shape is {}".format(dists.shape))
  dists = torch.from_numpy(dists)
#  print("dists is {}".format(dists))
  # calculate the sigmoid of the raw rgb values to constrain the values between 0 and 1
  rgb_vals = 1 / (1 + torch.exp(rgb_vol[..., :3]))
#  print("rgb_vals is {}".format(rgb_vals))
  # calculate the amount of light contributed by each section
  #  1 - exp(-sigma * dist) <- sigma is the volume density vol
  #
  #  NOTE: the sigma is the 4th entry of rgb_vol
  #   Since alpha is strictly between 0 and 1 -> need to apply ReLU to the sigma
  relu = torch.nn.ReLU()
  alpha = 1.0 - torch.exp(-relu(rgb_vol[...,3]) * dists)
#  print("alpha is {}".format(alpha))
  # Now need to calculate weights -> which is the amount of light blocked earlier along
  #  the ray -> this is a rolling product from j=1 to i-1
  # Need to do cumulative product on 1-alpha + 1e-10
  prev = 1.0
  cumprod = torch.full((1,), prev, dtype=torch.float64)
  for col in range(1, alpha.shape[0]):
    prev *= (1-alpha[col-1] + 1e-10)
    cumprod = torch.vstack([cumprod, prev])
#  print("cumprod is {}".format(cumprod))
  weights = alpha[..., None] * cumprod
#  print("weights is {}".format(weights))
  # now that we have the weights and the alpha for each rgb - multiply them all together and sum the vector to output a single rgb
  rgb = torch.sum(weights * rgb_vals, axis=-2)
#  print("rgb is {}".format(rgb))
  
  return rgb
