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
        img = img.resize((128, 28))
        # img = img.resize((94, 24))
        img = ToTensor()(img)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        img = Normalize(mean, std)(img)

        label = self.imgs[index].stem.split('_')[0]
        label = label.upper()
        label = ''.join(ch for ch in label if ch.isalnum())
        length = len(label)
        # TODO: Move to the collate_fn with torch padding
        # for i in range(len(label) - 1):
        #     if label[i] == label[i + 1]:
        #         label = label[: i + 1] + '-' + label[i + 1 :]

        while len(label) < config.SEQ_LEN:
            label += '-'
        label = [config.CHARS_DICT[char] for char in label]
        label = torch.tensor(label)

        return img, label, length


# def collate_fn(batch):
#     imgs = []
#     labels = []
#     lengths = []
#     for _, sample in enumerate(batch):
#         img, label, length = sample
#         imgs.append(torch.from_numpy(img))
#         labels.extend(label)
#         lengths.append(length)
#     labels = np.asarray(labels).flatten().astype(np.int)

#     return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


if __name__ == '__main__':
    lpd = LicensePlateDataset('data2')
    img, label, length = next(iter(lpd))
    print(img.size())
    print(img.mean())
    print(img.std())
    print(label)
    print(length)
    print(len(lpd))
    # from torch.utils.data import DataLoader

    # dl = DataLoader(lpd, batch_size=2)
    # print(len(dl))
    # for sample in dl:
    #     print(sample)
    #     break
