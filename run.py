from typing import List
import argbind
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from datasets import train, test, PairedSlakh
from model import Matcher
import torch


def accuracy(est, act):
    return np.sum(est == act) / len(act)

@argbind.bind()
def run(
    spectrogram_hidden_size: int = 500,
    midi_hidden_size: int = 500,
    num_layers: int = 3,
    spectrogram_kernels: List[int] = None,
    midi_kernels: List[int] = None,
    spectrogram_channels: List[int] = None,
    midi_channels: List[int] = None,
    dropout: float = .1,
    lr: float = .003,
    batch_size: int = 50,
    epochs: int = 100,
    device: str = 'cuda'
):
    n_freq = 512 // 2 + 1
    n_midi = 88
    if spectrogram_kernels is None:
        spectrogram_kernels = [10, 5, 3]
    if midi_kernels is None:
        midi_kernels = [10, 5, 3]
    if spectrogram_channels is None:
        spectrogram_channels = [4, 8, 16]
    if midi_channels is None:
        midi_channels = [4, 8, 16]
    model = Matcher(
        n_freq,
        n_midi,
        spectrogram_hidden_size,
        midi_hidden_size,
        num_layers,
        dropout,
        spectrogram_kernels,
        midi_kernels,
        spectrogram_channels,
        midi_channels,
        device
    )
    best_acc = 0
    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = BCELoss(reduction='mean')
    train_dataset = PairedSlakh(train)
    test_dataset = PairedSlakh(test)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for batch in dataloader:
            optimizer.zero_grad()
            spectogram, piano_roll, act = (
                batch["spectrogram"].to(device),
                batch["piano_roll"].to(device),
                batch["same"].to(device)
            )
            est = model(spectogram, piano_roll)
            loss = loss_func(est, act)
            print(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
            break

    estimated = []
    actual = []
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for batch in test_loader:
        with torch.no_grad():
            spectogram, piano_roll, act = (
                batch["spectrogram"].to(device),
                batch["piano_roll"].to(device),
                batch["same"].to(device)
            )
            actual.append(act[0].item())
            estimated.append(
                model(spectogram, piano_roll)[0].item()
            )
    print(f"Test Accuracy: {accuracy(np.array(actual), np.array(estimated))}")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        run()