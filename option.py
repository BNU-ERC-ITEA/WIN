import argparse, torch

parser = argparse.ArgumentParser(description='Well-posed Invertible Network for Reversible Image Conversion')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--GPUs', type=str, default=[0], help='GPUs id')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../Datasets/', help='dataset directory')
parser.add_argument('--task', type=str, default='hiding', help='RIC task: hiding | decolorization | rescaling')
parser.add_argument('--method', type=str, default='WIN_Naive', help='method name: WIN_Naive | WIN')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1 for hiding or colorization | 2 or 4 for rescaling')
parser.add_argument("--save_img", default=False, action="store_true", help='save images in testing mode')

# Model specifications
parser.add_argument('--num_secrets', type=int, default=1, help='number of secrets: only for image hiding tasks')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='test', help='train | test | complexity')

args = parser.parse_args()
