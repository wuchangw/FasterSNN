import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    def __init__(self, root_dir, target_size=(64, 64, 64), time_steps=4):
        self.root_dir = root_dir
        self.target_size = target_size
        self.time_steps = time_steps
        self.classes = ['AD', 'MCI', 'CN']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(('.nii', '.nii.gz')):
                        samples.append((os.path.join(class_dir, file_name), self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_data = nib.load(img_path).get_fdata()
        img_data = self._resize_3d(img_data)
        img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-5)

        time_sequence = []
        for _ in range(self.time_steps):
            perturbed = img_data + np.random.normal(0, 0.1, size=img_data.shape)
            time_sequence.append(perturbed)

        img_tensor = torch.FloatTensor(np.array(time_sequence))  # [T, H, W, D]
        return img_tensor, label  # Return shape [T, H, W, D]

    def _resize_3d(self, img):
        result = np.zeros(self.target_size)
        result[:min(img.shape[0], result.shape[0]),
        :min(img.shape[1], result.shape[1]),
        :min(img.shape[2], result.shape[2])] = \
            img[:result.shape[0], :result.shape[1], :result.shape[2]]
        return result