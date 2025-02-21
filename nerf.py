#!/usr/bin/python

import os
import sys
import numpy as np
from load_blender import *
from training_models import *
from display_nerf import *
import model_utils
from opts import *
import validate_utils
import json
from datetime import datetime

fileName = os.path.abspath(__file__)
fileDir = os.path.dirname(fileName)

model_mdata = os.path.join(fileDir, 'models/models_metadata.json')
models_dir = os.path.join(fileDir, 'models/')

torch.autograd.set_detect_anomaly(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# define execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# TODO - also parametrize the model as well
# API to load the saved model file given the parameters of the input to test such as:
# - model file name (if given)
# - number of positional encodings
# - is viewdir enabled
# - number of view encodings
def get_model_file_given_inputs(pos_enc, viewdir, view_enc, N_pts):
  err = float('inf')
  model_file = None

  with open(model_mdata) as mfile:
    mJSON = json.load(mfile)
    for mdataEntry in mJSON.get('model_files', {}):
      if pos_enc == mdataEntry.get('pos_enc', 0) and \
         viewdir == mdataEntry.get('viewdir', True) and \
         view_enc == mdataEntry.get('view_enc', 0) and \
         N_pts == mdataEntry.get('N_pts', 64) and \
         err < mdataEntry.get('err', float('inf')):
        model_file = mdataEntry.get('model_file')
        err = mdataEntry.get('err')
  
  return model_file

# given a model file get the params that would determine input size 
# (pos_enc, viewdir, viewdir_enc)
def get_model_params(model_file_name):
  with open(model_mdata) as mfile:
    mJSON = json.load(mfile)
    for mdataEntry in mJSON.get('model_files', {}):
      if model_file_name == mdataEntry.get('model_file'):
        return mdataEntry.get('pos_enc'), mdataEntry.get('view_dir'), mdataEntry.get('view_enc'), mdataEntry.get('N_pts')
  return None, None, None, None

def update_mdata_file(mdata_entry):
  with open(model_mdata, 'r') as mfile:
    mdataJSON = json.load(mfile)
  
  if not mdataJSON.get('model_files'):
    mdataJSON['model_files'] = [mdata_entry]
  else:
    mdataJSON['model_files'].append(mdata_entry)
  with open(model_mdata, 'w') as mfile:
    json.dump(mdataJSON, mfile, ensure_ascii=False, indent=4)

def load_train_blender(near, far, imgScale):
  # for now hardcode the example for lego
  trainJSONFile = os.path.join(fileDir, "nerf_synthetic/lego/transforms_train.json")
  blenderObj = LoadBlender(trainJSONFile, imgScale, near, far)
  imgObjects = blenderObj.rtrvImgObjects()
  return imgObjects

# train using images in blender format where the image intrinsics are provided in
# JSON format
def train_blender_data(model_file, imgScale, epochs, near, far, pos_enc, view_dir, view_enc, N_pts):

  trainImgObjs = load_train_blender(near, far, imgScale)

  model_mdata = {'from_model_file': model_file, 
                 'pos_enc': pos_enc,
                 'view_dir': view_dir,
                 'view_enc': view_enc,
                 "N_pts": N_pts}
  
  train_model = defaultNet(input_dir_dim=(3 if view_dir else 0), pos_enc=pos_enc)
  if model_file:
    train_model.load_state_dict(torch.load(model_file, weights_only=True)) 

  for epoch in range(epochs):
    print("epoch {}".format(epoch))
    img_idx = np.random.randint(len(trainImgObjs))
    train_data = trainImgObjs[img_idx].genModelData()
    print("train_data = {}".format(train_data[:3, :]))
    print(train_data.shape)

    loss = model_utils.train(epoch, model=train_model,
                             training_data=train_data,
                             batch_size=1024*5,
                             N_points=N_pts,
                             pos_enc=pos_enc,
                             view_dir=view_dir,
                             view_enc=view_enc)
    
    if (epoch % 100 == 99 and epoch > 0) or \
       epoch == epochs - 1:
      epoch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
      model_file_name = os.path.join(models_dir, "model_{}_{}".format(epoch_timestamp, epoch))
      mdata_entry = {'model_file': model_file_name, 'loss': loss.item(), 'epoch': epoch}
      mdata_entry.update(model_mdata)
      update_mdata_file(mdata_entry)
      torch.save(train_model.state_dict(), model_file_name)

  return train_model

# dataset in tiny_nerf_data.npz contains 100 training images 
# that are 100 by 100 - much smaller than usual
#
# load one image and its poses
def train_tiny_nerf_data(model_file, epochs, near, far, pos_enc, view_dir, view_enc, N_pts):
  data = np.load('tiny_nerf_data.npz')
  images = torch.Tensor(data['images'])
  poses = torch.Tensor(data['poses'])
  focal = torch.Tensor(data['focal'])
  H, W = images.shape[1:3]
  print(images.shape, poses.shape, focal)

  model_mdata = {'from_model_file': model_file, 
                 'pos_enc': pos_enc,
                 'view_dir': view_dir,
                 'view_enc': view_enc,
                 "N_pts": N_pts}

  train_model = defaultNet(input_dir_dim=(3 if view_dir else 0), pos_enc=pos_enc)
  if model_file:
    train_model.load_state_dict(torch.load(model_file, weights_only=True))

  for epoch in range(epochs):
    print("epoch {}".format(epoch))
    img_idx = np.random.randint(0, 100)
    train_data = model_utils.get_image_data(img_idx, images, poses, focal, near, far)
    print("train_data = {}".format(train_data[:3, :]))
    print(train_data.shape)

    loss = model_utils.train(epoch, model=train_model,
                             training_data=train_data,
                             batch_size=1024*5,
                             N_points=N_pts,
                             pos_enc=pos_enc,
                             view_dir=view_dir,
                            view_enc=view_enc)
    
    if (epoch % 100 == 99 and epoch > 0) or \
       epoch == epochs - 1:
      epoch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
      model_file_name = os.path.join(models_dir, "model_{}_{}".format(epoch_timestamp, epoch))
      mdata_entry = {'model_file': model_file_name, 'loss': loss.item(), 'epoch': epoch}
      mdata_entry.update(model_mdata)
      update_mdata_file(mdata_entry)
      torch.save(train_model.state_dict(), model_file_name)

  return train_model

def interactive_tiny_nerf_data(model_file_name):
  data = np.load('tiny_nerf_data.npz')
  images = torch.Tensor(data['images'])
  focal = torch.Tensor(data['focal'])
  H, W = images.shape[1:3]

  model_file = os.path.join(models_dir, model_file_name)
  pos_enc, view_dir, view_enc, N_pts = get_model_params(model_file)

  interactive_model = defaultNet(input_dir_dim=(3 if view_dir else 0), pos_enc=pos_enc)
  interactive_model.load_state_dict(torch.load(model_file, weights_only=True))

  interactive_mode(interactive_model, H, W, focal, 2., 6., N_pts, pos_enc, view_dir, view_enc)

def video_tiny_nerf_data(model_file_name):
  data = np.load('tiny_nerf_data.npz')
  images = torch.Tensor(data['images'])
  focal = torch.Tensor(data['focal'])
  H, W = images.shape[1:3]

  model_file = os.path.join(models_dir, model_file_name)
  pos_enc, view_dir, view_enc, N_pts = get_model_params(model_file)
  print("{}. {}, {}, {}".format(pos_enc, view_dir, view_enc, N_pts))
  video_model = defaultNet(input_dir_dim=(3 if view_dir else 0), pos_enc=pos_enc)
  video_model.load_state_dict(torch.load(model_file, weights_only=True))

  generate_video(video_model, H, W, focal, 2., 6., N_pts, pos_enc, view_dir, view_enc)

def main():
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  args = get_opts()
  if len(args.validate) > 0 and os.path.isfile(os.path.join(models_dir, args.validate)):
    validate_model = defaultNet()
    validate_model_file = os.path.join(models_dir, args.validate)
    print("loading model parameters from file {}".format(validate_model_file))
    validate_model.load_state_dict(torch.load(validate_model_file, weights_only=True))
    pos_enc, view_dir, view_enc, N_pts = get_model_params(validate_model_file)
    print("{}, {}, {}, {}".format(pos_enc, view_dir, view_enc, N_pts))
    if pos_enc is None or view_dir is None or view_enc is None or N_pts is None:
      print("ERROR - missing model parameters in model metadata file for model file {}".format(args.validate))
      return
    validate_utils.validate_tiny_nerf_data(validate_model, 
                                           near=args.near, far=args.far,
                                           pos_enc=pos_enc, 
                                           view_dir=view_dir, 
                                           view_enc=view_enc,
                                           N_points=N_pts)
  elif len(args.interactive) and os.path.isfile(os.path.join(models_dir, args.interactive)):
    interactive_tiny_nerf_data(args.interactive)
  elif len(args.video) and os.path.isfile(os.path.join(models_dir, args.video)):
    video_tiny_nerf_data(args.video)
  else:
    viewdir_flag = not args.no_view_dir
    if len(args.load_file) and \
      (os.path.isfile(os.path.join(models_dir, args.load_file)) or args.load_file == 'best'):
      # look for the model file with the least error given the pos_enc, 
      if args.load_file == 'best':
        model_file = get_model_file_given_inputs(args.pos_enc, viewdir_flag, args.viewdir_enc, args.num_pts)
        pos_enc, view_dir, view_enc, N_pts = args.pos_enc, viewdir_flag, args.view_enc, args.num_pts
      else:
        model_file = os.path.join(models_dir, args.load_file)
        pos_enc, view_dir, view_enc, N_pts = get_model_params(model_file)
        if not pos_enc or not view_dir or not view_enc or not N_pts:
          print("ERROR - missing model parameters in model metadata file for model file {}".format(model_file))
    else:
       model_file = None
       pos_enc, view_dir, view_enc, N_pts = args.pos_enc, viewdir_flag, args.view_enc, args.num_pts
      
    train_model = train_tiny_nerf_data(model_file=model_file, epochs=args.epochs,
                                       near=args.near, far=args.far,
                                       pos_enc=pos_enc, 
                                       view_dir=view_dir, 
                                       view_enc=view_enc,
                                       N_pts=N_pts)
   # train_model = train_blender_data(model_file=model_file,
   #                                  imgScale=4.0,
   #                                  epochs=args.epochs,
   #                                  near=0.,
   #                                  far=1.,
   #                                  pos_enc=pos_enc,
   #                                  view_dir=view_dir,
   #                                  view_enc=view_enc,
   #                                  N_pts=N_pts)
    validate_utils.validate_tiny_nerf_data(train_model,
                                           near=args.near, far=args.far,
                                           pos_enc=pos_enc, 
                                           view_dir=view_dir, 
                                           view_enc=view_enc,
                                           N_points=N_pts)
    #validate_utils.validate_blender_data(rootDir=fileDir,
    #                                     val_model=train_model,
    #                                     imgScale=4.0,
    #                                     near=0.,
    #                                     far=1.,
    #                                     pos_enc=pos_enc,
    #                                     view_dir=view_dir,
    #                                     view_enc=view_enc,
    #                                     N_points=N_pts)
  
if __name__ == '__main__':
  main()
