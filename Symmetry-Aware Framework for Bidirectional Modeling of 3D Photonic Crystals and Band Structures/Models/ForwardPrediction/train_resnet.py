import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Models.ForwardPrediction.CNNNetWork import CombinedNetwork
import copy
import subprocess
from data.dataset_test import SavedDatasetTest

out_dir = "Forward_Model"
saved_dataset_dir = "data/saved_dataset"
dataset = SavedDatasetTest(saved_dataset_dir)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
random_seed = 100

train_data, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)

_num_workers = min(2, max(0, (os.cpu_count() or 1) - 1))
_use_pin_memory = torch.cuda.is_available()

train_dataloader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    pin_memory=_use_pin_memory,
    num_workers=_num_workers,
    persistent_workers=bool(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=128,
    drop_last=True,
    pin_memory=_use_pin_memory,
    num_workers=_num_workers,
    persistent_workers=bool(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)


def get_available_gpus():
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        return []

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)

        available_gpus = []
        lines = result.stdout.strip().split('\n')
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 3:
                gpu_id = int(parts[0])
                memory_used = int(parts[1])
                memory_total = int(parts[2])

                # GPU is considered free if memory usage is < 80%
                memory_usage_ratio = memory_used / memory_total
                if memory_usage_ratio < 0.8:
                    available_gpus.append(gpu_id)
                    print(f"GPU {gpu_id}: Memory usage {memory_usage_ratio:.1%} - Available")
                else:
                    print(f"GPU {gpu_id}: Memory usage {memory_usage_ratio:.1%} - Busy")

        return available_gpus

    except Exception as e:
        print(f"Could not get GPU info: {e}")
        return list(range(torch.cuda.device_count()))


# Get available GPUs
print("Detecting available GPUs...")
available_gpus = get_available_gpus()

# Multi-GPU configuration
max_gpus = 3
use_multi_gpu = True

if not available_gpus:
    device = torch.device("cpu")
    selected_gpus = None
else:
    if use_multi_gpu and len(available_gpus) > 1:
        selected_gpus = available_gpus[:max_gpus]
        device = torch.device(f"cuda:{selected_gpus[0]}")  # Main device
    else:
        selected_gpus = [available_gpus[0]]
        device = torch.device(f"cuda:{selected_gpus[0]}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

use_resnet = True
use_kpoint = True
use_lattice = False
use_enhanced_kpoint = True

model = CombinedNetwork(use_resnet=use_resnet, use_kpoint=use_kpoint, use_lattice=use_lattice,
                        use_enhanced_kpoint=use_enhanced_kpoint)

if selected_gpus and len(selected_gpus) > 1:
    print(f"Using Multi-GPU training: {selected_gpus}")
    model = torch.nn.DataParallel(model, device_ids=selected_gpus)
    model = model.to(device)
elif selected_gpus and len(selected_gpus) == 1:
    print(f"Using Single-GPU training: {selected_gpus[0]}")
    model = model.to(device)
else:
    print("Using CPU training")
    model = model.to(device)
print(model)

loss_fn = nn.HuberLoss(delta=0.01)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

os.makedirs(out_dir, exist_ok=True)
epoch = 250


def _remove_module_prefix(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


for i in range(epoch):
    print(f"------- Epoch {i + 1} Start -------")
    model.train()
    total_train_loss = 0
    for batch_idx, batch_data in enumerate(train_dataloader):
        data, ka_features, a_matrices, epsvalue, targets, _index, ka_mask, label_mask = batch_data

        if torch.cuda.is_available():
            data, ka_features, a_matrices, epsvalue, targets, label_mask = [x.to(device) for x in
                                                                            [data, ka_features, a_matrices,
                                                                             epsvalue, targets, label_mask]]

        outputs = model(data, ka_features, a_matrices)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    print(f"Epoch {i + 1}/{epoch}, Train Loss: {total_train_loss / len(train_dataloader):.6f}")

# Training complete, save the final model
save_path = os.path.join(out_dir, "CNN_model_final.pth")
current_sd = copy.deepcopy(model.state_dict())
final_model_state_dict = _remove_module_prefix(current_sd)
final_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

torch.save({
    'epoch': epoch,
    'model_state_dict': final_model_state_dict,
    'optimizer_state_dict': final_optimizer_state_dict,
    'use_data_parallel': isinstance(model, torch.nn.DataParallel),
}, save_path)

# Use the current model (final state) for testing
model.eval()
total_test_loss = 0
test = []
prediction = []
test_masks = []
with torch.no_grad():
    for batch_data in test_dataloader:
        # Unpack data with mask (only one index)
        data, ka_features, a_matrices, epsvalue, targets, _index, ka_mask, label_mask = batch_data

        if torch.cuda.is_available():
            data, ka_features, a_matrices, epsvalue, targets, label_mask = [x.to(device) for x in
                                                                            [data, ka_features, a_matrices,
                                                                             epsvalue, targets, label_mask]]

        outputs = model(data, ka_features, a_matrices)

        # Use masked loss
        loss = loss_fn(outputs, targets)

        test.append(targets)
        prediction.append(outputs)
        test_masks.append(label_mask)
        total_test_loss += loss.item()

    test = torch.cat(test, dim=0)
    prediction = torch.cat(prediction, dim=0)
    test_masks = torch.cat(test_masks, dim=0)
    print(f"Test Loss: {total_test_loss / len(test_dataloader):.10f}")

    torch.save(test, os.path.join(out_dir, "test_data.pt"))
    torch.save(prediction, os.path.join(out_dir, "Prediction.pt"))
    torch.save(test_masks, os.path.join(out_dir, "test_mask.pt"))