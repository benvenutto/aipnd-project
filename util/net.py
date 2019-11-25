from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Architectures
ARCHITECTURES = dict(vgg19=(models.vgg19_bn, 'classifier', 25088),
                     resnet152=(models.resnet152, 'fc', 2048),
                     densenet201=(models.densenet201, 'classifier', 1920),
                     inception_v3=(models.inception_v3, 'fc', 2048),
                     resnext101=(models.resnext101_32x8d, 'fc', 2048))

def get_architectures():
    return ARCHITECTURES.keys()

def make_reproduceable(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_model(arch, out_features, hidden_units, dropout, leaky_relu):

    # Create model
    model_ref, classifier_name, in_features = ARCHITECTURES[arch]
    image_size = 224
    if arch == 'inception_v3':
        model = model_ref(pretrained=True, aux_logits=False)
        image_size = 299
    else:
        model = model_ref(pretrained=True)

    # Freeze model features
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Inject new classifier
    new_classifier = _make_classifier(in_features, out_features, hidden_units, dropout, leaky_relu)
    setattr(model, classifier_name, new_classifier)

    # Save classifier features
    model._classifier_parameters = {
        'out_features': out_features,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'leaky_relu': leaky_relu
    }
    return model, new_classifier, image_size

def _make_classifier(in_features, out_features, num_hidden_layers, p_dropout, leaky_relu):
    layers = OrderedDict()
    num_out = in_features
    for n in range(1, num_hidden_layers + 1):
        num_in = num_out
        f = 2 ** (int(math.log2(num_in)) - 1)
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

    classifier = nn.Sequential(layers)
    for name, param in classifier.named_parameters():
        if len(param.shape) < 2:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.kaiming_uniform_(param)
    return classifier

def train(train_dl, model, criterion, optimiser, device=torch.device('cpu')):
    model = model.train()
    running_loss = 0

    for images, labels in train_dl:
        images, labels = images.to(device=device), labels.to(device=device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    else:
        return running_loss / len(train_dl)

def eval(eval_dl, model, criterion, device=torch.device('cpu')):
    model = model.eval()
    running_loss = 0
    running_accuracy = 0

    for images, labels in eval_dl:
        images, labels = images.to(device=device), labels.to(device=device)

        with torch.no_grad():
            outputs = model(images)
            raw_proba = torch.exp(outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_labels = torch.argmax(raw_proba, dim=1)
            accuracy = torch.sum(pred_labels == labels) / float(pred_labels.shape[0])
            running_accuracy += accuracy
    else:
        return running_accuracy / len(eval_dl), running_loss / len(eval_dl)
