# this is opts file that sets the available options for executing nerf
import argparse

def get_opts():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--train', action='store_true', default=True,
                      help='run nerf in training mode')
  parser.add_argument('--epochs', type=int, default=1,
                      help='set number of epochs to train image')
  parser.add_argument('--validate', type=str, default='',
                      help='run the nerf model passed in the file in validation mode')
  parser.add_argument('--pos_enc', type=int, default=6,
                       help="set number of positive encodings for input")
  
  
  return parser.parse_args()
