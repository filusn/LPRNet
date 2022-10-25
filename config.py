import os
import pathlib
import random

import numpy as np
import torch
from torchvision import transforms

DEBUG = False

if DEBUG:
    torch.autograd.set_detect_anomaly(True)

TRAIN_DATASET = ''
VAL_DATASET = ''
TEST_DATASET = ''

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SEED = 10
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

MODELS_PATH = pathlib.Path('saved_models')
if not MODELS_PATH.exists():
    MODELS_PATH.mkdir()

# fmt: off
CHARS = [
    '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z',
]
# fmt: on

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
BLANK_SIGN = '-'
INPUT_LEN = 30

# TODO: Include normalization.
# TODO: Include augmentations for training on real dataset.
transforms_train = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
