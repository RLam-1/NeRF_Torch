# This file contains the APIs to train the model

import torch
from torch.optim import Adam
from render import *
from ray_utils import *

def get_RGB_point_from_ray(model, batch_entry, N_points):
  # each row contains [ray_o <x,y,z>, ray_d <x,y,z>, near, far, pixel_color <r,g,b,a>]
  # each entry is ray_orig, ray_dir -> need to calculate pts for each ray
  ray_o, ray_d = batch_entry[..., 0:3], batch_entry[..., 3:6]
  near, far = batch_entry[..., 6:8]
  pts, zvals = gen_ray_pts(ray_o, ray_d, near, far, N_points)
  # append the normalized direction to the pts entries
  viewdir = ray_d / np.linalg.norm(ray_d)
  viewdir = np.broadcast_to(viewdir, (pts.shape[0], viewdir.shape[0]))
#  print("viewdir shape is {}".format(viewdir.shape))
  train_in = torch.from_numpy(np.concatenate([pts, viewdir], axis=-1))
#  print("train_in shape is {}".format(train_in.shape))
  # get rgb and volume density value for the given ray using the pts and the viewdir
  # 
  rgb_vol_vals = model(train_in)
        
  # get the rgb of that ray using classical volumetric rendering
  rgb = volumetric_render(rgb_vol_vals, zvals, ray_d)
  
  return rgb
  
def get_RGB_points_from_batch(model, batch_data, N_points):
  # each row contains [ray_o <x,y,z>, ray_d <x,y,z>, near, far, pixel_color <r,g,b,a>]
  # each entry is ray_orig, ray_dir -> need to calculate pts for each ray
  ray_o, ray_d = batch_data[..., 0:3], batch_data[..., 3:6]
  near, far = batch_data[..., 6], batch_data[..., 7]
  batch_pts, z_vals = None, None
  for row in range(batch_data.shape[0]):
    pts, zvals = gen_ray_pts(ray_o[row], ray_d[row], near[row], far[row], N_points)
    # append the normalized direction to the pts entries
    viewdir = ray_d[row] / np.linalg.norm(ray_d[row])
    viewdir = np.broadcast_to(viewdir, (pts.shape[0], viewdir.shape[0]))
 #   print("viewdir shape is {}".format(viewdir.shape))
    train_in = torch.from_numpy(np.concatenate([pts, viewdir], axis=-1))
    zvals = torch.from_numpy(zvals)
 #   print("train_in shape is {}".format(train_in.shape))
    if batch_pts is None:
      batch_pts = train_in
    else:
      batch_pts = torch.vstack([batch_pts, train_in])
      
    if z_vals is None:
      z_vals = zvals
    else:
      z_vals = torch.vstack([z_vals, zvals])
  
  batch_pts = torch.reshape(batch_pts, (batch_data.shape[0], N_points, batch_pts.shape[-1]))
  print("batch_pts shape is {}".format(batch_pts.shape))
  # get rgb and volume density value for the given ray using the pts and the viewdir
  # 
  batch_rgb_vol_vals = model(batch_pts)
  model_output = None
  for entry in range(batch_rgb_vol_vals.shape[0]):
    # get the rgb of that ray using classical volumetric rendering
    rgb = volumetric_render(batch_rgb_vol_vals[entry,:,:], z_vals[entry], ray_d[entry])
    if model_output is None:
      model_output = rgb
    else:
      model_output = torch.vstack([model_output, rgb])
      
  return model_output 
  
def train_batch_entry_separately(model, batch_data, N_points):
  model_output = None
  for batch_entry in batch_data:
    # get the rgb of that ray using classical volumetric rendering
    rgb = get_RGB_point_from_ray(model, batch_entry, N_points)
  #  print("rgb is {}".format(rgb))
    # add this rgb value (of the ray) to model output so we can calculate loss
    if model_output is None:
      model_output = rgb
    else:
      model_output = torch.vstack([model_output, rgb])
      
  return model_output
  
def train(epochs, model, training_data, batch_size, N_points):
  # keep track of best accuracy - if model more accurate than best accuracy save params of that model
  best_accuracy = 0.0
  
  # TODO parametrize the optimizer
  loss_function = torch.nn.MSELoss()
  optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
  
  # define execution device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # convert model params and buffers to CPU or Cuda
  model.to(device)
  model.train(True)
  
  for epoch in range(epochs):
    batch_start = 0 
    while batch_start < training_data.shape[0]:
      batch_data = training_data[batch_start:batch_start + batch_size, :]
 #     model_output = train_batch_entry_separately(model, batch_data, N_points)
      model_output = get_RGB_points_from_batch(model, batch_data, N_points)
      
      # get the expected output of the batch - just drop the 'a' in rgba for now
      expected_output = torch.from_numpy(batch_data[:, -4:-1])
   #   print("expected_output is {}".format(expected_output))
   #   print("expected_output shape is {}".format(expected_output.shape))
   #   print("model_output is {}".format(model_output))
   #   print("model_output shape is {}".format(model_output.shape))
      # zero the parameter gradients
      optimizer.zero_grad()
      
      # calculate loss based on model_output vs expected_output
      loss = loss_function(model_output, expected_output)
      # do backprop on the loss
      loss.backward()      
      # adjust parameters based on calculated gradients
      optimizer.step()
      
      print("loss item is {}".format(loss.item()))

      batch_start = batch_start + batch_size + 1
