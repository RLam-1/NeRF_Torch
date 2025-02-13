#!/usr/bin/python

import os
import sys
import numpy as np
from load_blender import *
from training_models import *
import model_utils
from opts import *
import validate_utils

fileName = os.path.abspath(__file__)
fileDir = os.path.dirname(fileName)

torch.autograd.set_detect_anomaly(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# define execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def load_blender():
  # for now hardcode the example for lego
  trainJSONFile = os.path.join(fileDir, "nerf_synthetic/lego/transforms_train_mini_3.json")
  blenderObj = LoadBlender(trainJSONFile)
  imgObjects = blenderObj.rtrvImgObjects()
  
  train_data = None
  # now get the model data from the imgObjects and combine them all into one list
  for imgObj in imgObjects:
    model_data = imgObj.genModelData()
    if train_data is None:
      train_data = model_data
    else:
      train_data = np.concatenate([train_data, model_data])
  
  return train_data

# dataset in tiny_nerf_data.npz contains 100 training images 
# that are 100 by 100 - much smaller than usual
#
# load one image and its poses
def train_tiny_nerf_data(epochs=1, near=2, far=6, pos_enc=6):
  data = np.load('tiny_nerf_data.npz')
  images = torch.Tensor(data['images'])
  poses = torch.Tensor(data['poses'])
  focal = torch.Tensor(data['focal'])
  H, W = images.shape[1:3]
  print(images.shape, poses.shape, focal)

  model_file = model_utils.get_latest_model_file("models")
  train_model = defaultNet(pos_enc=pos_enc)

  if model_file is not None:
    print("loading model parameters from file {}".format(model_file))
   # train_model.load_state_dict(torch.load(model_file, weights_only=True))

  for epoch in range(epochs):
    print("epoch {}".format(epoch))
    img_idx = np.random.randint(0, 100)
    train_data = model_utils.get_image_data(img_idx, images, poses, focal, near, far)
    print("train_data = {}".format(train_data[:3, :]))
    print(train_data.shape)
  
    # shuffle the data for training
    #np.random.shuffle(train_data)

    model_utils.train(epoch, model=train_model, training_data=train_data, batch_size=1024*5, N_points=64, pos_enc=pos_enc)

  return train_model
  
def main():
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  args = get_opts()
  if len(args.validate) > 0 and os.path.isFile(args.validate):
    validate_model = defaultNet()
    print("loading model parameters from file {}".format(args.validate))
    #  validate_model.load_state_dict(torch.load(args.validate, weights_only=True))

    validate_utils.validate_tiny_nerf_data(validate_model, pos_enc=args.pos_enc)

  elif args.train:
    print(args.epochs)
    train_model = train_tiny_nerf_data(epochs=args.epochs, pos_enc=args.pos_enc)
    validate_utils.validate_tiny_nerf_data(train_model, pos_enc=args.pos_enc)
  
if __name__ == '__main__':
  main()
