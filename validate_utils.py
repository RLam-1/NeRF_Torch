# this file contains the APIs used to validate the model
import numpy as np
import model_utils
import torch
import matplotlib.pyplot as plt
import os
from load_blender import *

def load_val_blender(rootDir, near, far, imgScale):
  valJSONFile = os.path.join(rootDir, "nerf_synthetic/lego/transforms_val.json")
  blenderObj = LoadBlender(valJSONFile, imgScale, near, far)
  imgObjects = blenderObj.rtrvImgObjects()
  return imgObjects

def validate_blender_data(rootDir, val_model, imgScale, near, far, pos_enc, view_dir, view_enc, N_points):
  imgObjects = load_val_blender(rootDir, near, far, imgScale)
  img_idx = np.random.randint(len(imgObjects))

  validate_data, _ = imgObjects[img_idx].genModelData()

  def loss_function(output, target):
    loss = torch.mean(torch.square(output - target))
    return loss

  val_model.eval()
  val_model.cuda()

  avg_loss = 0.0

  with torch.no_grad():
    batch_start, batch_count = 0, 0
    # from empirical data - was able to fit 1024 * 5 input points onto 8GB VRAM GPU
    batch_size = 1024 * 5
    v_output = torch.Tensor(device='cuda')
    while batch_start < validate_data.shape[0]:
      v_batch_data = validate_data[batch_start:batch_start + batch_size, :]
      v_batch_output = model_utils.get_RGB_points_from_batch(val_model, v_batch_data, N_points, pos_enc, view_dir, view_enc, training=False)
      expected_batch_output = validate_data[batch_start:batch_start + batch_size, -3:]
      v_batch_loss = loss_function(v_batch_output, expected_batch_output)
      print("validate loss is {}".format(v_batch_loss))
      avg_loss += v_batch_loss.item()
      v_output = torch.concat([v_output, v_batch_output])

      batch_start += batch_size
      batch_count += 1

  avg_loss /= batch_count
  print("near = {}, far = {}".format(near, far))

  # reshape the expected output and model output so that it takes on image shape
  img_shape = list(imgObjects[img_idx].getImgRGB().shape)
  v_output_img = torch.reshape(v_output, img_shape).cpu()
  #expected_output_img_from_structure = torch.reshape(validate_data[:, -3:], img_shape).cpu()
  expected_output_img = imgObjects[img_idx].image.cpu()
  print(torch.nonzero(v_output_img))

  plt.figure(figsize=(10,4))
  plt.subplot(121)
  plt.imshow(v_output_img)
  plt.title("model output")
  plt.subplot(122)
  plt.imshow(expected_output_img)
  plt.title("expected output")
  plt.show()

def validate_tiny_nerf_data(val_model, near, far, pos_enc, view_dir, view_enc, N_points):
  # load the tiny nerf data
  data = np.load('tiny_nerf_data.npz')
  images = torch.Tensor(data['images'])
  poses = torch.Tensor(data['poses'])
  focal = torch.Tensor(data['focal'])
  H, W = images.shape[1:3]
  print(images.shape, poses.shape, focal)

  img_idx = np.random.randint(100, 103)

  validate_data = model_utils.get_image_data(img_idx, images, poses, focal, near, far)

  def loss_function(output, target):
    loss = torch.mean(torch.square(output - target))
    return loss

  val_model.eval()
  val_model.cuda()

  with torch.no_grad():
    v_output = model_utils.get_RGB_points_from_batch(val_model, validate_data, N_points, pos_enc, view_dir, view_enc, training=False)
    expected_output = validate_data[:, -3:]
    v_loss = loss_function(v_output, expected_output)
    print("validate loss is {}".format(v_loss))

  # reshape the expected output and model output so that it takes on image shape
  v_output_img = torch.reshape(v_output, list(images.shape[1:])).cpu()
  expected_output_img = torch.reshape(expected_output, list(images.shape[1:])).cpu()
  print(torch.nonzero(v_output_img))

  plt.figure(figsize=(10,4))
  plt.subplot(121)
  plt.imshow(v_output_img)
  plt.title("model output")
  plt.subplot(122)
  plt.imshow(expected_output_img)
  plt.title("expected output")
  plt.show()
