import argparse
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json
from util import net, state
from torch import nn, optim

# Some constants

ARCHITECTURES = dict(vgg19=(models.vgg19_bn, 'classifier', 25088),
                     resnet152=(models.resnet152, 'fc', 2048),
                     densenet201=(models.densenet201, 'classifier', 1920),
                     inception_v3=(models.inception_v3, 'fc', 2048),
                     resnext101=(models.resnext101_32x8d, 'fc', 2048))

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, action='store', help='specify the data directory')
parser.add_argument('--save_dir', type=str, action='store', default='./', help='checkpoint save directory')
parser.add_argument('--arch', type=str, action='store', default='vgg19', choices=list(ARCHITECTURES.keys()), help='CNN architecture')
parser.add_argument('--learning_rate', type=float, action='store', default='0.003', help='learning rate')
parser.add_argument('--hidden_units', type=int, action='store', default='1', help='number of hidden units')
parser.add_argument('--epochs', type=int, action='store', default='3', help='number of epochs')
parser.add_argument('--gpu', action='store_const', const=True, default=False, help='Enable GPU')

parser.add_argument('--leaky_relu', action='store_const', const=True, default=False, help='Use Leaky ReLU')
parser.add_argument('--dropout', type=float, action='store', default='0.5', help='Probability of dropout')
parser.add_argument('--seed', type=int, action='store', default=None, help='Random seed for reproduceable results')
parser.add_argument('--batch_size', type=int, action='store', default='128', help='Mini-batch size')
parser.add_argument('--num_threads', type=int, action='store', default='1', help='Dataloader threads')

args = parser.parse_args()

# Configure CPU/GPU
compute_device = torch.device('cpu')
if args.gpu and torch.cuda.is_available():
    compute_device = torch.device('cuda:0')
print(f'Will use device {compute_device} for computation')

