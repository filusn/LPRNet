from itertools import groupby

import torch
import torch.nn as nn
import torch.optim as optim
from Levenshtein import distance
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
        # if loss == torch.inf:
        #     continue
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
    num_infs = 0

    for idx, (imgs, labels, lengths) in enumerate(loop):
        imgs, labels, lengths = (
            imgs.to(config.DEVICE),
            labels.to(config.DEVICE),
            lengths.to(config.DEVICE),
        )

        with torch.no_grad():
            logits = model(imgs)

            # Calculation of loss
            input_lengths, target_lengths = sparse_tuple_for_ctc(16, lengths)
            log_probs = logits.permute(2, 0, 1)
            log_probs = log_probs.log_softmax(2)

            batch_loss = loss_fn(
                log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths
            )

            if not batch_loss == torch.inf:
                loss += batch_loss.item()
            else:
                num_infs += 1

            # Calculation of accuracy
            logits = torch.argmax(logits, axis=1)
            logits = logits.cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()

            for encoded_pred, encoded_label in zip(logits, labels):
                # Remove consecutive duplicates
                encoded_pred = [key for key, _ in groupby(encoded_pred)]
                # Remove blank characters
                encoded_pred = [ch for ch in encoded_pred if ch != len(config.CHARS) - 1]
                encoded_label = [ch for ch in encoded_label if ch != len(config.CHARS) - 1]

                if encoded_pred == encoded_label:
                    true_pos += 1

                # Calculation of Levenshtein distance
                if levenshtein:
                    decoded_pred = ''.join([config.CHARS[ch] for ch in encoded_pred])
                    decoded_label = ''.join([config.CHARS[ch] for ch in encoded_label])
                    levenshtein_dist += distance(decoded_pred, decoded_label)

    mean_loss = loss / len(loader.dataset)
    accuracy = true_pos / len(loader.dataset) * 100
    print(len(loader), true_pos, num_infs)
    mean_levenshtein = (
        round(levenshtein_dist / len(loader.dataset), 2) if levenshtein else 'not calculated'
    )

    print(
        f'[INFO] Evaluation mean loss / accuracy / mean Levenshtein distance -> {mean_loss:.2f} / {accuracy:.2f} / {mean_levenshtein}'
    )


def train():
    train_dataset = LicensePlateDataset('data')
    train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)

    lprnet = LPRNet(len(config.CHARS)).to(config.DEVICE)
    # lprnet.load_state_dict(torch.load(config.MODELS_PATH / 'lprnet_50.pth'))

    opt = optim.RMSprop(lprnet.parameters(), lr=config.LEARNING_RATE)
    ctc_loss = nn.CTCLoss(blank=len(config.CHARS) - 1, reduction='mean')

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        train_epoch(lprnet, train_dataloader, opt, ctc_loss)
        if (epoch + 1) % 10 == 0:
            evaluate(lprnet, train_dataloader, ctc_loss, levenshtein=True)

        if (epoch + 1) % 50 == 0:
            torch.save(lprnet.state_dict(), config.MODELS_PATH / f'lprnet_{epoch + 1}.pth')

    # lprnet.load_state_dict(torch.load(config.MODELS_PATH / 'lprnet_50.pth'))
    # evaluate(lprnet, train_dataloader, ctc_loss, levenshtein=True)


if __name__ == '__main__':
    train()
