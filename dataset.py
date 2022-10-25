import pathlib

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor

import config


class LicensePlateDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = pathlib.Path(path)
        self.imgs = list(self.path.iterdir())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = img.resize((128, 28))

        # TODO: Move to the augmentations for train/test ds.
        img = ToTensor()(img)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        img = Normalize(mean, std)(img)

        label = self.imgs[index].stem.split('_')[0]
        label = label.upper()

        # Checking for alphanumeric characters.
        # May be removed with a proper dataset.
        label = ''.join(ch for ch in label if ch.isalnum())
        length = len(label)

        # TODO: Move to the collate_fn with torch padding
        while len(label) < config.INPUT_LEN:
            label += '-'
        label = [config.CHARS_DICT[char] for char in label]
        label = torch.tensor(label)

        return img, label, length
