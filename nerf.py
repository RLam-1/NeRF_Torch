#!/usr/bin/python

import os
import sys
import numpy as np
from load_blender import *
from training_models import *
import model_utils

fileName = os.path.abspath(__file__)
fileDir = os.path.dirname(fileName)

torch.autograd.set_detect_anomaly(True)

def train():
  # for now hardcode the example for lego
  trainJSONFile = os.path.join(fileDir, "nerf_synthetic/lego/transforms_train_mini.json")
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
      
  print(train_data[0])
  print(train_data.shape)
  
  train_model = defaultNet(depth=2)
  model_utils.train(epochs=1, model=train_model, training_data=train_data, batch_size=100, N_points=64)
  
def main():
  train()
  
if __name__ == '__main__':
  main()
