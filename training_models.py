# This file contains the training models used to generate the volume density and view-dependent gradients

import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
class defaultNet(nn.Module):
  def __init__(self, depth=8, width=256, input_pos_dim=3, input_dir_dim=3, output_dim=4, skips=[4], pos_enc=6):

    super(defaultNet, self).__init__()
    self.depth = depth
    self.width = width
    self.input_pos_dim = input_pos_dim + input_pos_dim * 2 * pos_enc
    self.input_dir_dim = input_dir_dim
    self.output_dim = output_dim
    self.skips = skips
    self.input_dim = self.input_pos_dim + self.input_dir_dim
    self.activation_function = nn.ReLU()
  
    # first layer is linear with input = input_pos_dim + input_dir_dim
    self.layers = nn.ModuleList()
    self.layers.append(torch.nn.Linear(self.input_dim, self.width))

    for i in range(1, depth):
      # if layer is skip layer where the input is fed in with the output from the previous layer need to expand the input dimension
      if i in skips:
        input_size = self.input_dim + self.width
      else:
        input_size = self.width
        
      self.layers.append(torch.nn.Linear(input_size, self.width))
    
    # final output layer with no activation
    self.layers.append(torch.nn.Linear(self.width, self.output_dim))
   # self.double()
    
  def forward(self, x):
    # first forward layer
    print("input shape is {}".format(x.shape))
    hidden_o = self.activation_function(self.layers[0](x))
    
    for i in range(1, self.depth):
      if i in self.skips:
        _input = torch.cat([hidden_o, x], -1)
      else:
        _input = hidden_o
        
      hidden_o = self.activation_function(self.layers[i](_input))

    output = self.layers[-1](hidden_o)
    return output
      
 
    
