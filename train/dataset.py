import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        root_dir = os.path.expanduser(root_dir)
        self.samples = []
        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(folder_path, fname)
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        self.samples.append((path, label))
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"{e} : {path}")
                        os.remove(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[Error loading] {e}: {path}")
            raise e
        return img, label


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.subset)



