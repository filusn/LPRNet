import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LicensePlateDataset
from model import LPRNet


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def train_epoch(model, loader, opt, loss_fn):
    loop = tqdm(loader, leave=True)
    model.train()
    loss_epoch = 0

    for idx, (imgs, labels, lengths) in enumerate(loop):
        imgs, labels, lengths = (
            imgs.to(config.DEVICE),
            labels.to(config.DEVICE),
            lengths.to(config.DEVICE),
        )

        input_lengths, target_lengths = sparse_tuple_for_ctc(18, lengths)

        logits = model(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        opt.zero_grad()
        loss = loss_fn(
            log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths
        )
        loss.backward()
        opt.step()

        loss_epoch += loss.item()

    print(f'Epoch loss: {loss_epoch}')


def train():
    train_dataset = LicensePlateDataset('data')
    train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)

    lprnet = LPRNet(len(config.CHARS)).to(config.DEVICE)
    opt = optim.RMSprop(lprnet.parameters(), lr=config.LEARNING_RATE)
    ctc_loss = nn.CTCLoss(blank=len(config.CHARS) - 1, reduction='mean')

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        train_epoch(lprnet, train_dataloader, opt, ctc_loss)


if __name__ == '__main__':
    train()
