import argparse
import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
import math, json

ARCHS = dict(vgg19=(models.vgg19_bn, 'classifier', 25088),
             resnet152=(models.resnet152, 'fc', 2048),
             densenet201=(models.densenet201, 'classifier', 1920),
             inception_v3=(models.inception_v3, 'fc', 2048),
             resnext101=(models.resnext101_32x8d, 'fc', 2048))

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, action='store', help='specify the data directory')
parser.add_argument('--save_dir', type=str, action='store', default='./', help='checkpoint save directory')
parser.add_argument('--arch', type=str, action='store', default='vgg19', choices=list(ARCHS.keys()), help='CNN architecture')
parser.add_argument('--learning_rate', type=float, action='store', default='0.003', help='learning rate')
parser.add_argument('--hidden_units', type=int, action='store', default='1', help='number of hidden units')
parser.add_argument('--epochs', type=int, action='store', default='3', help='number of epochs')
parser.add_argument('--gpu', action='store_const', const=torch.device('cuda:0'), default=torch.device('cpu'), help='Enable GPU')
parser.add_argument('--dropout', type=float, action='store', default='0.5', help='Probability of dropout')
parser.add_argument('--leaky_relu', action='store_const', const=True, default=False, help='Use Leaky ReLU')

args = parser.parse_args()

model_ref, classifier_name, in_features = ARCHS[args.arch]
model = model_ref(pretrained=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
out_fetures = len(cat_to_name)

def make_classifier(in_features, out_features, num_hidden_layers, p_dropout, leaky_relu):
    layers = OrderedDict()
    num_out = in_features
    for n in range(1, num_hidden_layers + 1):
        num_in = num_out
        f = 2**(int(math.log2(num_in)) - 1)
        if f >= out_features:
            num_out = f
        else:
            num_out = out_features
        layers[f'fc{n}'] = nn.Linear(num_in, num_out)
        layers[f'bn{n}'] = nn.BatchNorm1d(num_out)
        if leaky_relu is True:
            layers[f'prelu{n}'] = nn.PReLU()
        else:
            layers[f'relu{n}'] = nn.ReLU()
        if p_dropout > 0.0:
            layers[f'dropout{n}'] = nn.Dropout(p=p_dropout)
    layers[f'fc-final'] = nn.Linear(num_out, out_features)
    layers['softmax'] = nn.LogSoftmax(dim=1)
    return nn.Sequential(layers)

new_classifier = make_classifier(in_features, out_fetures, args.hidden_units, args.dropout, args.leaky_relu)
setattr(model, classifier_name, new_classifier)
print(model)
