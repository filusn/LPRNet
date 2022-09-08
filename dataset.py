import pathlib
from re import L

from PIL import Image
import torch
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
        img = img.resize((94, 24))
        img = ToTensor()(img)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        img = Normalize(mean, std)(img)

        label = self.imgs[index].stem.split('_')[0]
        label = label.upper()
        label = ''.join(ch for ch in label if ch.isalnum())

        while len(label) < 9:
            splitted = False
            for i in range(len(label) - 1):
                if label[i] == label[i + 1]:
                    label = label[: i + 1] + '-' + label[i + 1 :]
                    splitted = True
                    break
            if not splitted:
                label += '-'
        label = [config.CHARS_DICT[char] for char in label]
        label = torch.tensor(label)

        return img, label, len(label)


if __name__ == '__main__':
    lpd = LicensePlateDataset('data')
    img, label, length = next(iter(lpd))
    print(img.size())
    print(img.mean())
    print(img.std())
    print(label)
    print(length)
    print(len(lpd))
    from torch.utils.data import DataLoader

    dl = DataLoader(lpd)
    print(len(dl))
