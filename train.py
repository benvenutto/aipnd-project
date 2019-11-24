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

NUM_THREADS = 8
BATCH_SIZE = 128
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

args = parser.parse_args()

# Configure CPU/GPU
compute_device = torch.device('cpu')
if args.gpu and torch.cuda.is_available():
    compute_device = torch.device('cuda:0')

# Set reproduceable random seed, if specified
if args.seed is not None:
    net.make_reproduceable(args.seed)

# Load class descriptions
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
out_fetures = len(cat_to_name)

# Load selected pre-trained model
model_ref, classifier_name, in_features = ARCHITECTURES[args.arch]
model = model_ref(pretrained=True)

# Freeze model features
for name, param in model.named_parameters():
    param.requires_grad = False

# Inject new classifier
new_classifier = net.make_classifier(in_features, out_fetures, args.hidden_units, args.dropout, args.leaky_relu)
setattr(model, classifier_name, new_classifier)
model = model.to(device=compute_device)

# Setup data directories
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# Data transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

eval_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Data sets
train_dataset = ImageFolder(root=train_dir,
                            transform=train_transforms)

valid_dataset = ImageFolder(root=valid_dir,
                            transform=eval_transforms)

# Data loaders
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_THREADS,
                              pin_memory=args.gpu)

valid_dataloader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_THREADS,
                              pin_memory=args.gpu)

# Setup loss function and optimiser
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

# Run training epochs
best_valid_loss = float('inf')

for e in range(args.epochs):
    train_loss = net.train(train_dataloader, model,
                           criterion=criterion,
                           optimiser=optimiser,
                           device=compute_device)
    valid_accuracy, valid_loss = net.eval(valid_dataloader, model,
                                          criterion=criterion,
                                          device=compute_device)
    if valid_loss < best_valid_loss:
        state.save_snapshot(args.arch, model, optimiser, e, train_dataset.class_to_idx)
        best_valid_loss = valid_loss

    print(f'Epoch {e + 1} - training loss={train_loss:.6f}' \
          f', validation accuracy={valid_accuracy:.6f} and loss={valid_loss:.6f}')

print(model)
