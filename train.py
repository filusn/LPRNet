from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Levenshtein import distance
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LicensePlateDataset
from model import LPRNetEU


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), torch.from_numpy(np.array(lengths)))


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

        input_lengths = torch.ones((len(lengths)), dtype=torch.int) * config.INPUT_LEN

        logits = model(imgs)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        opt.zero_grad()
        loss = loss_fn(log_probs, labels, input_lengths=input_lengths, target_lengths=lengths)
        loss.backward()
        opt.step()

        loss_epoch += loss.item()

    print(f'Epoch loss: {loss_epoch}')


def evaluate(model, loader, loss_fn, levenshtein=False):
    loop = tqdm(loader, leave=True)
    model.eval()

    loss = 0
    true_pos = 0
    levenshtein_dist = 0

    for idx, (imgs, labels, lengths) in enumerate(loop):
        imgs, labels, lengths = (
            imgs.to(config.DEVICE),
            labels.to(config.DEVICE),
            lengths.to(config.DEVICE),
        )

        with torch.no_grad():
            logits = model(imgs)

            # Calculation of loss
            input_lengths = torch.ones((len(lengths)), dtype=torch.int) * config.INPUT_LEN
            log_probs = logits.permute(2, 0, 1)
            log_probs = log_probs.log_softmax(2)

            batch_loss = loss_fn(
                log_probs, labels, input_lengths=input_lengths, target_lengths=lengths
            )

            loss += batch_loss.item()

            # Calculation of accuracy
            logits = torch.argmax(logits, axis=1)
            logits = logits.cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()

            for encoded_pred, encoded_label in zip(logits, labels):
                # Remove consecutive duplicates
                encoded_pred = [key for key, _ in groupby(encoded_pred)]
                # Remove blank characters
                encoded_pred = [
                    ch for ch in encoded_pred if ch != config.CHARS.index(config.BLANK_SIGN)
                ]
                encoded_label = [
                    ch for ch in encoded_label if ch != config.CHARS.index(config.BLANK_SIGN)
                ]

                if encoded_pred == encoded_label:
                    true_pos += 1

                # Calculation of Levenshtein distance
                if levenshtein:
                    decoded_pred = ''.join([config.CHARS[ch] for ch in encoded_pred])
                    decoded_label = ''.join([config.CHARS[ch] for ch in encoded_label])
                    levenshtein_dist += distance(decoded_pred, decoded_label)

    mean_loss = loss / len(loader.dataset)
    accuracy = true_pos / len(loader.dataset) * 100
    mean_levenshtein = (
        round(levenshtein_dist / len(loader.dataset), 2) if levenshtein else 'not calculated'
    )

    print(
        f'[INFO] Evaluation loss / mean loss / accuracy / mean Levenshtein distance -> {loss:2f} / {mean_loss:.5f} / {accuracy:.4f} / {mean_levenshtein}'
    )

    return loss, mean_loss, accuracy, mean_levenshtein


def train():
    train_dataset = LicensePlateDataset(config.TRAIN_DATASET)
    train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)

    val_dataset = LicensePlateDataset(config.VAL_DATASET)
    val_dataloader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False)

    test_dataset = LicensePlateDataset(config.TEST_DATASET)
    test_dataloader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False)

    lprnet = LPRNetEU(len(config.CHARS)).to(config.DEVICE)

    # TODO: Check other optimizers.
    opt = optim.RMSprop(lprnet.parameters(), lr=config.LEARNING_RATE)
    # TODO: Change the scheduler for some good one.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt)
    ctc_loss = nn.CTCLoss(blank=config.CHARS_DICT[config.BLANK_SIGN], reduction='mean')

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        train_epoch(lprnet, train_dataloader, opt, ctc_loss)
        val_loss, val_mean_loss, _, _ = evaluate(
            lprnet, val_dataloader, ctc_loss, levenshtein=True
        )
        scheduler.step(val_mean_loss)

        # TODO: Change saving for the best/last weights.
        if (epoch + 1) % 10 == 0:
            torch.save(lprnet.state_dict(), config.MODELS_PATH / f'lprnet_{epoch + 1}.pth')

    # TEST
    lprnet.load_state_dict(torch.load(config.MODELS_PATH / 'lprnet_best.pth'))
    loss, mean_loss, accuracy, mean_levenshtein = evaluate(
        lprnet, test_dataloader, ctc_loss, levenshtein=True
    )


if __name__ == '__main__':
    train()
