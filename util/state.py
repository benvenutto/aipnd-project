import torch
from util import net

SNAPSHOT_FILE = "snapshot.pickle"

def save_snapshot(arch, model, optimiser, shortest_side, image_size, epoch, class_to_idx):
    snapshot = {
        'architecture': arch,
        'model': model.state_dict(),
        'classifier_parameters': model._classifier_parameters,
        'shortest_side': shortest_side,
        'image_size': image_size,
        'optimiser': optimiser.state_dict(),
        'epoch': epoch,
        'class_to_idx': class_to_idx
    }
    torch.save(snapshot, f'{arch}-{SNAPSHOT_FILE}')

def load_snapshot(checkpoint_file, device):
    snapshot = torch.load(checkpoint_file, map_location=device)
    arch = snapshot['architecture']
    classifier_params = snapshot['classifier_parameters']
    model, _, shortest_side, image_size = net.make_model(arch, **classifier_params)
    model.load_state_dict(snapshot['model'])
    model = model.to(device=device)
    class_to_index = snapshot['class_to_idx']
    index_to_class = {v: k for k, v in class_to_index.items()}
    return model, shortest_side, image_size, class_to_index, index_to_class
