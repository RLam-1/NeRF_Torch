# This file contains the APIs for rendering

import numpy as np
import torch

# This API calculates the final rgb from the rgb_vol_density readings along a ray
def volumetric_render(rgb_vol, zvals, ray_dir):
  # first calculate the distances between the zvals of the points along the ray
  dists = zvals[..., 1:] - zvals[..., :-1]
  # need to add infinite for the distance of the last point
  dists = np.concatenate([dists, [1e10]], axis=-1)
  #dists = np.concatenate([dists, np.broadcast_to([1e10], (dists.shape[0]))], axis=-1)
  
  # multiply the distance by the norm of the ray dir to get the distance in world space
  dists *= np.linalg.norm(ray_dir, axis=-1)
  print("rgb_vol is {}".format(rgb_vol))
  print("dists shape is {}".format(dists.shape))
  dists = torch.from_numpy(dists)
  dists = dists.to('cuda')
  print("dists is {}".format(dists))
  # calculate the sigmoid of the raw rgb values to constrain the values between 0 and 1
  rgb_vals = 1 / (1 + torch.exp(rgb_vol[..., :3]))
  print("rgb_vals is {}".format(rgb_vals))
  # calculate the amount of light contributed by each section
  #  1 - exp(-sigma * dist) <- sigma is the volume density vol
  #
  #  NOTE: the sigma is the 4th entry of rgb_vol
  #   Since alpha is strictly between 0 and 1 -> need to apply ReLU to the sigma
  relu = torch.nn.ReLU()
  alpha = 1.0 - torch.exp(-relu(rgb_vol[...,3]) * dists)
  print("relu rgb_vol {}".format(relu(rgb_vol[...,3])))
  print("dists {}".format(dists))
  print("alpha is {}".format(alpha))
  # Now need to calculate weights -> which is the amount of light blocked earlier along
  #  the ray -> this is a rolling product from j=1 to i-1
  # Need to do cumulative product on 1-alpha + 1e-10
  prev = 1.0
  cumprod = torch.full((1,), prev, dtype=torch.float64)
  for col in range(1, alpha.shape[0]):
    prev *= (1-alpha[col-1] + 1e-10)
    cumprod = torch.vstack([cumprod, prev])
  print("cumprod is {}".format(cumprod))
  weights = alpha[..., None] * cumprod
  print("weights is {}".format(weights))
  # now that we have the weights and the alpha for each rgb - multiply them all together and sum the vector to output a single rgb
  rgb = torch.sum(weights * rgb_vals, axis=-2)
  print("rgb is {}".format(rgb))
  
  return rgb

# This API calculates the final rgb from the rgb_vol_density readings along a ray
def batch_volumetric_render(rgb_vol, zvals, rays_dir):
  # first calculate the distances between the zvals of the points along the ray
  dists = zvals[..., 1:] - zvals[..., :-1]
  # need to add infinite for the distance of the last point
  end = torch.full((dists.shape[0], 1), 1e10)
  dists = torch.cat([dists, end], dim=-1)
  #dists = np.concatenate([dists, np.broadcast_to([1e10], (dists.shape[0]))], axis=-1)
  
  # multiply the distance by the norm of the ray dir to get the distance in world space
  rays_dir_norms = torch.norm(rays_dir, dim=-1)
  rays_dir_norms = rays_dir_norms.reshape(rays_dir_norms.shape[-1], 1)
  dists = dists * rays_dir_norms
  print("rgb_vol is {}".format(rgb_vol))
  print("rgb_vol nonzero {}".format(torch.nonzero(rgb_vol)))
#  print("dists shape is {}".format(dists.shape))
  # before the shape of the dists is horiz(dists along ray), vert(rays in batch)
  #    now the shape of the dists is z(rays in batch) y(dists along ray)
  dists = dists.reshape(dists.shape[0], dists.shape[1], 1)
  #dists = torch.Tensor(dists, device='cuda')
  print("dists is {}".format(dists))
  print("dists nonzero {}".format(dists))
  # calculate the sigmoid of the raw rgb values to constrain the values between 0 and 1
  rgb_vals = torch.sigmoid(rgb_vol[:, :, :3])
  print("rgb_vals is {}".format(rgb_vals))
  print("rgb vals nonzero {}".format(torch.nonzero(rgb_vals)))
  # calculate the amount of light contributed by each section
  #  1 - exp(-sigma * dist) <- sigma is the volume density vol
  #
  #  NOTE: the sigma is the 4th entry of rgb_vol
  #   Since alpha is strictly between 0 and 1 -> need to apply ReLU to the sigma
  # take the last column of each rgb_vol entry -> this is the volumetric density
  if torch.any(torch.relu(rgb_vol[..., 3:4])) == True:
    density = rgb_vol[..., 3:4]
  else:
    density = torch.sigmoid(rgb_vol[..., 3:4])
  alpha = 1.0 - torch.exp(-(torch.relu(density)) * dists)
  print("relu rgb_vol {}".format(torch.relu(rgb_vol[...,3:4])))
  print("dists {}".format(dists))
  print("alpha is {}".format(alpha))
  # Now need to calculate weights -> which is the amount of light blocked earlier along
  #  the ray -> this is a rolling product from j=1 to i-1
  # Need to do cumulative product on 1-alpha + 1e-10
  cumprod = 1.0 - alpha + 1e-10
  cumprod = torch.cumprod(cumprod, axis=-2)
  # NOTE that the rolling product ends at j-1 meaning that the first entry is 1
  # and the last element is removed
  cumprod = cumprod[:,:-1,:]
  firstelem = torch.ones((cumprod.shape[0],1,1), dtype=torch.float64)
  cumprod = torch.cat([firstelem, cumprod], axis=1)  
  print("cumprod is {}".format(cumprod))
  weights = alpha * cumprod
  print("weights is {}".format(weights))
  print("weights nonzero is {}".format(torch.nonzero(weights)))
#  print("weights is {}".format(weights))
  # now that we have the weights and the alpha for each rgb - multiply them all together and sum the vector to output a single rgb
  rgb = torch.sum(weights * rgb_vals, axis=-2)
  print("rgb is {}".format(rgb))
  print("rgb nonzero {}".format(torch.nonzero(rgb)))
  
  return rgb

def calculate_batch_positive_encoding(data, encode_len=6):
  encoded = [data]
  for i in range(encode_len):
    encoded.append(torch.sin(2. ** i * data))
    encoded.append(torch.cos(2. ** i * data))
  return torch.concatenate(encoded, -1)
