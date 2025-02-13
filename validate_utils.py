# this file contains the APIs used to validate the model
import numpy as np
import model_utils
import torch
import matplotlib.pyplot as plt 

def validate_tiny_nerf_data(val_model, near=2, far=6, N_points=64, pos_enc=6):
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
    v_output = model_utils.get_RGB_points_from_batch(val_model, validate_data, N_points, pos_enc)
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
