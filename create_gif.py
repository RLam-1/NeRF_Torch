import imageio
import os
import re

filename = os.path.abspath(__file__)
fileDir = os.path.dirname(filename)

framesDir = os.path.join(fileDir, 'video_frames')
angles = []
for frame in os.listdir(framesDir):
  # the name of the frame file is image_(angle in float)
  frameName = os.path.splitext(frame)[0]
  angles.append(float(frameName.split('_')[-1]))

# now sort the angles in increasing order
angles.sort()

images = []
for angle in angles:
  frameName = os.path.join(fileDir, 'video_frames/image_{}.png'.format(angle))
  images.append(imageio.imread(frameName))

imageio.mimsave('nerf.gif', images, fps=55)