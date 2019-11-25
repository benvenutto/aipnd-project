import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json
from util import net, state
from torch import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, action='store', help='data directory')
parser.add_argument('--save_dir', type=str, action='store', default='./', help='checkpoint save directory')
parser.add_argument('--arch', type=str, action='store', default='vgg19', choices=list(net.get_architectures()), help='CNN architecture')
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
print(f"Will use device '{compute_device}' for computation")

# Set reproduceable random seed, if specified
if args.seed is not None:
    net.make_reproduceable(args.seed)

# Load class descriptions
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
out_features = len(cat_to_name)

# Load selected pre-trained model
model, new_classifier, shortest_side, image_size = net.make_model(args.arch, out_features, args.hidden_units, args.dropout, args.leaky_relu)
model = model.to(device=compute_device)

# Setup data directories
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# Data transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(size=(image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

eval_transforms = transforms.Compose([
    transforms.Resize(size=shortest_side),
    transforms.CenterCrop(size=image_size),
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
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_threads,
                              pin_memory=args.gpu)

valid_dataloader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_threads,
                              pin_memory=args.gpu)

# Setup loss function and optimiser
criterion = nn.NLLLoss()
optimiser = optim.Adam(new_classifier.parameters(), lr=args.learning_rate)

# Run training epochs
best_valid_loss = float('inf')

for e in range(args.epochs):
    train_loss = net.train(train_dataloader,
                           model,
                           criterion=criterion,
                           optimiser=optimiser,
                           device=compute_device)
    valid_accuracy, valid_loss = net.eval(valid_dataloader,
                                          model,
                                          criterion=criterion,
                                          device=compute_device)
    if valid_loss < best_valid_loss:
        state.save_snapshot(args.arch, model, optimiser, shortest_side, image_size, e, train_dataset.class_to_idx)
        best_valid_loss = valid_loss

    print(f'Epoch {e + 1} - training loss={train_loss:.6f}' \
          f', validation accuracy={valid_accuracy:.6f} and loss={valid_loss:.6f}')

print('Training completed.')
