import argparse
import torch
import torchvision.models as models
import json
from util import net, state
from PIL import Image
import torchvision.transforms as transforms

# Some constants
ARCHITECTURES = dict(vgg19=(models.vgg19_bn, 'classifier', 25088),
                     resnet152=(models.resnet152, 'fc', 2048),
                     densenet201=(models.densenet201, 'classifier', 1920),
                     inception_v3=(models.inception_v3, 'fc', 2048),
                     resnext101=(models.resnext101_32x8d, 'fc', 2048))

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_file', type=str, action='store', help='image file')
parser.add_argument('checkpoint_file', type=str, action='store', help='checkpoint file')
parser.add_argument('--top_k', type=int, action='store', default='1', help='top K scores')
parser.add_argument('--category_names', type=str, action='store', help='file mapping categories to names')
parser.add_argument('--gpu', action='store_const', const=True, default=False, help='Enable GPU')

args = parser.parse_args()

# Configure CPU/GPU
compute_device = torch.device('cpu')
if args.gpu and torch.cuda.is_available():
    compute_device = torch.device('cuda:0')
print(f'Will use device {compute_device} for computation')

# Set reproduceable random seed, if specified
if args.seed is not None:
    net.make_reproduceable(args.seed)

# Load class descriptions
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
out_fetures = len(cat_to_name)

# Load checkpoint
model = state.load_snapshot(args.checkpoint_file)

# Open and process image
eval_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
pil_image = Image.open(args.image_file)
tensor = eval_transforms(pil_image)

