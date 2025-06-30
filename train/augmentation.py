from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BlipImageProcessor


class Transform:
    def __init__(self, model_name, augmentation=True):
        self.processor = BlipImageProcessor.from_pretrained(model_name)

        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    tuple(self.processor.size.values()),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.01
                ),
                transforms.RandomGrayscale(p=0.3)
            ])
        else:
            self.transform = None

    def __call__(self, x):
        if self.transform:
            x = self.transform(x)
        return self.processor(images=x, return_tensors='pt')["pixel_values"][0]


