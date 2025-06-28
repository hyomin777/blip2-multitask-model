from torchvision import transforms
from torchvision.transforms import AutoAugment, InterpolationMode


class ConservativeAugmentation:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.01
            ),
            transforms.RandomGrayscale(p=0.3)
        ])

    def __call__(self, x):
        return self.transform(x)

