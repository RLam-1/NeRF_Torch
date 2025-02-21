# This file contains the APIs to train the model

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from render import *
from ray_utils import *
from datetime import datetime
import glob
import time
import gc

def get_latest_model_file(models_path):
  model_files = glob.glob("{}/*".format(models_path))
  most_recent_file = None
  most_recent_time, highest_epoch = None, 0
  for model_file in model_files:
    _, timestamp, epoch = model_file.split('_')
    timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
    epoch = int(epoch)

    if most_recent_time is None or \
       timestamp > most_recent_time or \
       (timestamp == most_recent_time and epoch > highest_epoch):
      most_recent_file = model_file

  return most_recent_file

def get_rays_data(H, W, focal, pose, near, far):
  ray_dirs, ray_origs, _ = gen_img_rays(H, W, focal, pose)
  print("ray_dirs = {}, ray_origs = {}".format(ray_dirs[:3, :], ray_origs[:3, :]))
  near_far = torch.asarray([near, far])
  near_far = torch.broadcast_to(near_far, (ray_dirs.shape[0], near_far.shape[0]))
  rays_data = torch.concatenate([ray_origs, ray_dirs, near_far], dim=-1)
  return rays_data

def get_image_data(img_idx, images, poses, focal, near=2, far=6):
  H,W = images.shape[1:3]
  rays_data = get_rays_data(H, W, focal, poses[img_idx, :, :], near, far)
  img = images[img_idx, :, :, :].reshape(-1,3)
  print("img = {}".format(img[:3, :]))
  image_data = torch.concatenate([rays_data, img], dim=-1)
  print("image_data = {}".format(image_data[:3, :]))
  return image_data

def get_RGB_point_from_ray(model, batch_entry, N_points, view_dir):
  # each row contains [ray_o <x,y,z>, ray_d <x,y,z>, near, far, pixel_color <r,g,b,a>]
  # each entry is ray_orig, ray_dir -> need to calculate pts for each ray
  ray_o, ray_d = batch_entry[..., 0:3], batch_entry[..., 3:6]
  near, far = batch_entry[..., 6:8]
  pts, zvals = gen_ray_pts(ray_o, ray_d, near, far, N_points)
  # append the normalized direction to the pts entries
  if view_dir:
    viewdir = ray_d / np.linalg.norm(ray_d)
    viewdir = np.broadcast_to(viewdir, (pts.shape[0], viewdir.shape[0]))
    print("viewdir shape is {}".format(viewdir.shape))
    train_in = torch.from_numpy(np.concatenate([pts, viewdir], axis=-1))
  else:
    train_in = torch.from_numpy(pts)
  print("train_in shape is {}".format(train_in.shape))
  # get rgb and volume density value for the given ray using the pts and the viewdir
  # 
  rgb_vol_vals = model(train_in)
        
  # get the rgb of that ray using classical volumetric rendering
  rgb = volumetric_render(rgb_vol_vals, zvals, ray_d)
  
  return rgb
  
def get_RGB_points_from_batch(model, batch_data, N_points, pos_enc, view_dir, view_enc, training=True):
  # each row contains [ray_o <x,y,z>, ray_d <x,y,z>, near, far, pixel_color <r,g,b,a>]
  # each entry is ray_orig, ray_dir -> need to calculate pts for each ray
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  ray_o, ray_d = batch_data[..., 0:3], batch_data[..., 3:6]
  near, far = batch_data[..., 6:7], batch_data[..., 7:8]
  pts, z_vals = gen_pts_from_rays_batch(ray_o, ray_d, near, far, N_points)
  batch_pts = calculate_batch_positive_encoding(pts, pos_enc)
  # add the view dir
  # first need to reshape ray_d since the pts in the last 2 dimensions are from ONE RAY
  if view_dir:
    viewdir = ray_d / torch.norm(ray_d, dim=-1).reshape(ray_d.shape[0], 1)
    viewdir = viewdir.reshape(viewdir.shape[0], 1, viewdir.shape[-1])
    # plus need to project this viewdir along the rows dimension
    viewdir = torch.broadcast_to(viewdir, (viewdir.shape[0], pts.shape[-2], viewdir.shape[-1]))
    batch_pts = torch.cat((batch_pts, viewdir), dim=-1)

  print("batch_pts shape is {}".format(batch_pts.shape))
  # get rgb and volume density value for the given ray using the pts and the viewdir
  #
#  batch_pts_flat = torch.from_numpy(batch_pts)
  batch_pts_flat = batch_pts
  batch_pts_flat = torch.reshape(batch_pts_flat, [-1,batch_pts_flat.shape[-1]])
  #batch_pts_flat = batch_pts_flat.to(torch.double)
  batch_pts_flat = batch_pts_flat.to('cuda')
  start = time.time()
  batch_rgb_vol_vals = model(batch_pts_flat)
  batch_rgb_vol_vals = torch.reshape(batch_rgb_vol_vals, list(batch_pts.shape[:-1]) + [batch_rgb_vol_vals.shape[-1]])
  print("Elapsed time of model calc is {}".format(time.time() - start))
  model_output = batch_volumetric_render(batch_rgb_vol_vals, z_vals, ray_d, training=training)
      
  return model_output 
  
def train_batch_entry_separately(model, batch_data, N_points):
  model_output = None
  for batch_entry in batch_data:
    # get the rgb of that ray using classical volumetric rendering
    rgb = get_RGB_point_from_ray(model, batch_entry, N_points)
    print("rgb is {}".format(rgb))
    # add this rgb value (of the ray) to model output so we can calculate loss
    if model_output is None:
      model_output = rgb
    else:
      model_output = torch.vstack([model_output, rgb])
      
  return model_output
  
def train(epoch, model, training_data, batch_size, N_points, pos_enc, view_dir, view_enc):
  # keep track of best accuracy - if model more accurate than best accuracy save params of that model
  best_accuracy = 0.0
  model.train(True)
  model.cuda()
  # TODO parametrize the optimizer
  def loss_function(output, target):
    loss = torch.mean(torch.square(output - target))
    return loss
#  loss_function = nn.MSELoss()
  
  optimizer = Adam(model.parameters(), lr=5e-4)
  
  # add summary writer
  train_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  writer = SummaryWriter('runs/NeRF_trainer_{}'.format(train_timestamp))
  #training_data = torch.tensor(training_data, device='cuda')

  batch_start, batch_count = 0, 0
  batch_len = training_data.shape[0] / batch_size
  if training_data.shape[0] % batch_size > 0:
    batch_len += 1
  
  model_output = torch.Tensor(device='cuda')
  while batch_start < training_data.shape[0]:    
    print("batch number {}".format(batch_count))
    batch_data = training_data[batch_start:batch_start + batch_size, :]
    model_batch_output = get_RGB_points_from_batch(model, batch_data, N_points, pos_enc, view_dir, view_enc)
    model_output = torch.concat([model_output, model_batch_output])

    #model_output = model_batch_output
    print("model output nonzero values {}".format(torch.nonzero(model_output)))
    batch_start = batch_start + batch_size
    batch_count += 1

 # model_output = model_output.to(torch.double)
  expected_output = training_data[:, -3:]
#  expected_output = expected_output.to(torch.double)
  expected_output = expected_output.to('cuda')
  print("expected_output is {}".format(expected_output))
  print("expected_output shape is {}".format(expected_output.shape))
  print("model_output is {}".format(model_output))
  print("model_output shape is {}".format(model_output.shape))
      
  # zero the parameter gradients
  optimizer.zero_grad()
  # calculate loss based on model_output vs expected_output
  loss = loss_function(model_output, expected_output)
  # do backprop on the loss
  loss.backward()      
  # adjust parameters based on calculated gradients
  optimizer.step()

  print("loss is {}".format(loss))

  return loss

