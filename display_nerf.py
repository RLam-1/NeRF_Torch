import mediapy as media
import torch
import matplotlib.pyplot as plt
from ipywidgets import interactive, widgets
import model_utils
import imageio
import numpy as np
from tqdm import tqdm
import gc

# given this frame of coords
# y points up - xz coplanar with normal y

# first translate the camera along the z-axis
translate = lambda t: torch.Tensor([[1.,0,0,0],
                                    [0,1.,0,0],
                                    [0,0,1.,t],
                                    [0,0,0,1.]])

# then rotate around the x-axis 
rotateX = lambda phi: torch.Tensor([[1.,0,0,0],
                                    [0,np.cos(phi),-np.sin(phi),0],
                                    [0, np.sin(phi),np.cos(phi),0],
                                    [0,0,0,1.]
                                    ])

# then rotate around the y-axis
rotateY = lambda theta: torch.Tensor([[np.cos(theta),0,-np.sin(theta),0],
                                      [0,1.,0,0],
                                      [np.sin(theta),0,np.cos(theta),0],
                                      [0,0,0,1.]])

final_spherical_transform = torch.Tensor([[-1.,0,0,0],
                                          [0,0,1.,0],
                                          [0,1.,0,0],
                                          [0,0,0,1.]])
def get_spherical_pos(radius, phi, theta):
  c2w = translate(radius)
  c2w = rotateX(phi) @ c2w
  c2w = rotateY(theta) @ c2w
  c2w = final_spherical_transform @ c2w
  return c2w

def interactive_mode(model, H, W, focal, near, far, N_points, pos_enc, view_dir, view_enc):
  def f(**kwargs):
    pose = get_spherical_pos(**kwargs)
    ray_data = model_utils.get_rays_data(H, W, focal, pose, near, far)
    with torch.no_grad():
      i_output = model_utils.get_RGB_points_from_batch(model,
                                                       ray_data,
                                                       N_points,
                                                       pos_enc,
                                                       view_dir,
                                                       view_enc,
                                                       training=False)
    
    i_output_img = torch.reshape(i_output, (H, W, i_output.shape[-1])).cpu()
    i_output_img = torch.clamp(i_output_img, 0, 1)

    plt.figure(2, figsize=(20,6))
    plt.imshow(i_output_img)
    plt.show()

  sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
  )

  names = [
    ['radius', [4., 3., 5.]],
    ['phi', [-30., -90., 0]],
    ['theta', [100., 0., 360.]]
  ]

  #interactive_plot = interactive(f, **{s[0]: sldr(*s[1]) for s in names})
  interactive(f, **{s[0]: sldr(*s[1]) for s in names})
  #output = interactive_plot.children[-1]
  #output.layout_height = '350px'
  #interactive_plot


def generate_video(model, H, W, focal, near, far, N_points, pos_enc, view_dir, view_enc):
  frames = []
  for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = get_spherical_pos(4., -30., th)
    ray_data = model_utils.get_rays_data(H, W, focal, c2w, near, far)
    with torch.no_grad():
      i_output = model_utils.get_RGB_points_from_batch(model,
                                                       ray_data,
                                                       N_points,
                                                       pos_enc,
                                                       view_dir,
                                                       view_enc,
                                                       training=False)
    
    i_output_img = torch.reshape(i_output, (H, W, i_output.shape[-1])).cpu().to(torch.uint8)
    i_output_img = 255 * torch.clamp(i_output_img, 0, 1)
    frames.append(i_output_img)

  output_video = 'video.mp4'
  imageio.mimwrite(output_video, frames, fps=30, quality=7)
