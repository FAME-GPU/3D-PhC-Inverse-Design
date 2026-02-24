import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split


def safe_torch_load(path: str):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


class SavedDatasetTest(Dataset):
    def __init__(self, load_dir: str = "saved_dataset") -> None:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {load_dir}")

        file_paths = {
            'data': os.path.join(load_dir, "Tensor.pt"),
            'label': os.path.join(load_dir, "Label.pt"),
            'ka_features': os.path.join(load_dir, "K_a_features.pt"),
            'epsvalue': os.path.join(load_dir, "epsvalue.pt"),
            'a_matrices': os.path.join(load_dir, "A_matrices.pt"),
            'crystal_names': os.path.join(load_dir, "crystal_name.pt"),
            'metadata': os.path.join(load_dir, "metadata.pt")
        }

        required = ['data', 'label', 'ka_features', 'epsvalue', 'a_matrices', 'crystal_names']
        for k in required:
            if not os.path.exists(file_paths[k]):
                raise FileNotFoundError(f"Missing required file: {file_paths[k]}")

        self.data = safe_torch_load(file_paths['data'])
        self.label = safe_torch_load(file_paths['label'])
        self.ka_features = safe_torch_load(file_paths['ka_features'])
        self.epsvalue = safe_torch_load(file_paths['epsvalue'])
        self.a_matrices = safe_torch_load(file_paths['a_matrices'])
        self.crystal_names_lookup = safe_torch_load(file_paths['crystal_names'])

        self.metadata = {}
        if os.path.exists(file_paths['metadata']):
            self.metadata = safe_torch_load(file_paths['metadata'])

        # Optional metadata (used for mask calculation if present)
        self.original_ka_lengths = self.metadata.get('original_ka_lengths', None)
        self.original_label_lengths = self.metadata.get('original_label_lengths', None)
        self.sample_source_mapping = self.metadata.get('sample_source_mapping', None)

        # Unified max length (based on saved tensor dimensions)
        self.max_ka_features_len = int(self.ka_features.shape[1])
        self.max_label_len = int(self.label.shape[1])

        self.total_size = int(self.data.shape[0])

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        data = self.data[index]
        ka_features = self.ka_features[index]
        a_matrices = self.a_matrices[index]
        epsvalue = self.epsvalue[index]
        label = self.label[index]

        dataset_idx = self._get_original_dataset_idx(index)
        ka_mask = self._create_ka_mask_for_sample(index, dataset_idx)
        label_mask = self._create_label_mask_for_sample(index, dataset_idx, label)

        # Do not return dataset_idx to avoid confusion
        return data, ka_features, a_matrices, epsvalue, label, index, ka_mask, label_mask

    def _get_original_dataset_idx(self, sample_idx: int) -> int:
        if isinstance(self.sample_source_mapping, torch.Tensor):
            return int(self.sample_source_mapping[sample_idx].item())
        if isinstance(self.sample_source_mapping, (list, tuple)):
            return int(self.sample_source_mapping[sample_idx])
        # Default to 0 if no mapping exists
        return 0

    def _create_ka_mask_for_sample(self, sample_idx: int, dataset_idx: int) -> torch.Tensor:
        if self.original_ka_lengths is not None and len(self.original_ka_lengths) > dataset_idx:
            original_len = int(self.original_ka_lengths[dataset_idx])
        else:
            # Fallback: Estimate valid length from content (based on non-zero check)
            k = self.ka_features[sample_idx]
            # k shape: [T, 3]
            valid = (k.abs().sum(dim=-1) > 0)
            original_len = int(valid.nonzero().max().item() + 1) if valid.any() else 0

        mask = torch.zeros(self.max_ka_features_len, dtype=torch.bool)
        original_len = max(0, min(original_len, self.max_ka_features_len))
        mask[:original_len] = True
        return mask

    def _choose_label_length(self, dataset_idx: int, label_row: Optional[torch.Tensor]) -> int:
        # Target only allows two lengths
        allowed = (560, 920)

        # 1) Prioritize metadata
        if self.original_label_lengths is not None and len(self.original_label_lengths) > dataset_idx:
            meta_len = int(self.original_label_lengths[dataset_idx])
            if meta_len in allowed:
                return meta_len

        # 2) Heuristic: Prefer shorter length if the second half is zero
        if label_row is not None:
            T = int(label_row.shape[0])
            if T >= 920:
                tail_after_560_zero = bool(torch.all(label_row[560:] == 0))
                if tail_after_560_zero:
                    return 560
                return 920
            # If smaller than 920 but >= 560, choose 560
            if T >= 560:
                return 560

        # 3) Final fallback: Use the largest allowed length not exceeding max_label_len
        for v in sorted(allowed, reverse=True):
            if v <= self.max_label_len:
                return v
        return min(allowed)

    def _create_label_mask_for_sample(self, sample_idx: int, dataset_idx: int, label_row: torch.Tensor) -> torch.Tensor:
        target_len = self._choose_label_length(dataset_idx, label_row)
        mask = torch.zeros(self.max_label_len, dtype=torch.bool)
        target_len = max(0, min(target_len, self.max_label_len))
        mask[:target_len] = True
        return mask


def build_test_dataloader(load_dir: str = "saved_dataset",
                          batch_size: int = 32,
                          num_workers: Optional[int] = None,
                          pin_memory: Optional[bool] = None) -> Tuple[SavedDatasetTest, DataLoader]:
    dataset = SavedDatasetTest(load_dir)
    if num_workers is None:
        num_workers = min(2, max(0, (os.cpu_count() or 1) - 1))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dataset, dataloader


def build_train_test_dataloaders(load_dir: str = "saved_dataset",
                                 batch_size: int = 32,
                                 train_ratio: float = 0.8,
                                 seed: int = 100,
                                 num_workers: Optional[int] = None,
                                 pin_memory: Optional[bool] = None) -> Tuple[SavedDatasetTest, DataLoader, DataLoader, Subset, Subset]:
    dataset = SavedDatasetTest(load_dir)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    if num_workers is None:
        num_workers = min(2, max(0, (os.cpu_count() or 1) - 1))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataset, train_loader, test_loader, train_data, test_data


if __name__ == "__main__":
    ds, train_loader, test_loader, train_data, test_data = build_train_test_dataloaders()
    print(f"Loaded saved_dataset with {len(ds)} samples. Train={len(train_data)}, Test={len(test_data)}")
    print(f"Shapes: data={tuple(ds.data.shape)}, ka={tuple(ds.ka_features.shape)}, label={tuple(ds.label.shape)}")

    # Save 50 samples from the test set (save only global index, remove ds_idx field)
    os.makedirs("test_data", exist_ok=True)
    count = min(50, len(test_data))
    samples = []
    for i in range(count):
        data, ka, A, epsv, label, index,  ka_mask, label_mask = test_data[i]
        global_idx = int(index) if isinstance(index, (int,)) else int(index.item())
        samples.append(
            (
                data.cpu(),
                ka.cpu(),
                A.cpu(),
                epsv.cpu(),
                label.cpu(),
                global_idx,
                ka_mask.cpu(),
                label_mask.cpu(),
            )
        )

    out_path = os.path.join("test_data", "test_data.pt")
    torch.save(samples, out_path)
    print(f"Saved {len(samples)} test samples to {out_path}")