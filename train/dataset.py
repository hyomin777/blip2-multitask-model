import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, folder in enumerate(['positive', 'negative']):  # 0, 1
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
        img = Image.open(path).convert('RGB')
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



def pil_collate_fn(batch):
    """
    batch: list of dicts with keys 'image' (PIL.Image) and 'label' (int)
    """
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {'image': images, 'label': labels}



