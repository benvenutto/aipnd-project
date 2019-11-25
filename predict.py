import argparse
import torch
import json
from util import net, state
from PIL import Image
import torchvision.transforms as transforms

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_file', type=str, action='store', help='image file')
parser.add_argument('checkpoint_file', type=str, action='store', help='checkpoint file')
parser.add_argument('--top_k', type=int, action='store', default='1', help='top K scores')
parser.add_argument('--category_names', type=str, action='store', help='file mapping categories to names')
parser.add_argument('--gpu', action='store_const', const=True, default=False, help='Enable GPU')

parser.add_argument('--seed', type=int, action='store', default=None, help='Random seed for reproduceable results')

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
model, shortest_side, image_size, class_to_index, index_to_class = state.load_snapshot(args.checkpoint_file, device=compute_device)

# Open and process image
predict_transforms = transforms.Compose([
    transforms.Resize(size=shortest_side),
    transforms.CenterCrop(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def predict(image_path, transforms, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    image = Image.open(image_path)
    tensor = transforms(image)
    batch = tensor.unsqueeze(0).to(device=compute_device)
    with torch.no_grad():
        output = model(batch)
        raw_proba = torch.exp(output)
        topk_preds = raw_proba.topk(topk, dim=1)
        probs = topk_preds.values.cpu().squeeze().tolist()
        classes = topk_preds.indices.cpu().squeeze().tolist()
    return probs, classes

# Get predictions

probs, classes = predict(args.image_file, predict_transforms, model)

