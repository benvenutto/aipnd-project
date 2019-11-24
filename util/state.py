import torch

SNAPSHOT_FILE = "snapshot.pickle"

def save_snapshot(arch, model, optimiser, epoch, class_to_idx):
    snapshot = {
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'epoch': epoch,
        'class_to_idx': class_to_idx
    }
    torch.save(snapshot, f'{arch}-{SNAPSHOT_FILE}')

def load_snapshot(arch, model, optimiser, device):
    snapshot = torch.load(f'{arch}-{SNAPSHOT_FILE}', map_location=device)
    model.load_state_dict(snapshot['model'])
    optimiser.load_state_dict(snapshot['optimiser'])
    class_to_index = snapshot['class_to_idx']
    index_to_class = {v: k for k, v in class_to_index.items()}
    return snapshot['epoch'], class_to_index, index_to_class