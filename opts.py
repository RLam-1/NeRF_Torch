# this is opts file that sets the available options for executing nerf
import argparse

def get_opts():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--epochs', type=int, default=1,
                      help='set number of epochs to train image')
  parser.add_argument('--validate', type=str, default='',
                      help='run the nerf model passed in the file in validation mode')
  parser.add_argument('--pos_enc', type=int, default=6,
                       help="set number of positional encodings for input")
  parser.add_argument('--view_enc', type=int, default=0,
                      help='set number of directional encodings for input')
  parser.add_argument('--no_view_dir', action='store_true', default=False,
                      help='set whether to pass in viewdir for input')
  parser.add_argument('--load_file', type=str, default='',
                      help='pass in model file to train OR set \'best\' to pick file with least error')
  parser.add_argument('--num_pts', type=int, default=64,
                      help='number of pts to generate along rays')
  parser.add_argument('--near', type=int, default=2,
                      help='dimension of the near parameter for the pts along ray')
  parser.add_argument('--far', type=int, default=6,
                      help='dimension of the far parameter for the pts along ray')
  parser.add_argument('--interactive', type=str, default='',
                      help='run interactive mode on saved model passed as input')
  parser.add_argument('--video', type=str, default='',
                      help='generate video of saved model passed as input')
  parser.add_argument('--imgScale', type=float, default=2.0,
                      help='factor to which to scale down the image')
  
  return parser.parse_args()
