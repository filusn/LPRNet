import torch
from torchvision import transforms


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

# fmt: off
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', '-'
]
# fmt: on

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

# TODO: Calculate mean and std of all images
transforms_train = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize()
    ]
)
