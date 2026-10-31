"""Graph representations used by the exploratory polarization experiments."""
from pathlib import Path
import csv

import numpy as np
import torch
from torch import nn


def permute_within_graphs(sizes, rng):
    """Corrupt node features without mixing measurements from different ROIs."""
    offsets = np.cumsum([0, *sizes])
    return np.concatenate([rng.permutation(n) + offsets[i] for i, n in enumerate(sizes)])


class DGIEncoder(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.first = nn.Linear(inputs, 32)
        self.second = nn.Linear(32, 16)
        self.discriminator = nn.Parameter(torch.empty(16, 16))
        nn.init.xavier_uniform_(self.discriminator)

    def encode(self, features, adjacency):
        hidden = torch.relu(self.first(torch.sparse.mm(adjacency, features)))
        return self.second(torch.sparse.mm(adjacency, hidden))


def dgi_embedding(features, adjacency, graphs, seed, epochs, device, folder):
    """Train shared encoder with a separate summary and corruption per ROI."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    coo = adjacency.tocoo()
    a = torch.sparse_coo_tensor(np.stack([coo.row, coo.col]), coo.data,
                                size=coo.shape, device=device).coalesce()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    sizes = [n for n, _ in graphs]
    model = DGIEncoder(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        permutation = torch.as_tensor(permute_within_graphs(sizes, rng), device=device)
        optimizer.zero_grad()
        positive = model.encode(x, a)
        negative = model.encode(x[permutation], a)
        summaries = torch.stack([torch.sigmoid(part.mean(dim=0)) for part in positive.split(sizes)])
        node_summaries = torch.repeat_interleave(
            summaries, torch.as_tensor(sizes, device=device), dim=0)
        positive_score = ((positive @ model.discriminator) * node_summaries).sum(dim=1)
        negative_score = ((negative @ model.discriminator) * node_summaries).sum(dim=1)
        loss = .5 * (nn.functional.softplus(-positive_score).mean()
                     + nn.functional.softplus(negative_score).mean())
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite DGI loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        history.append((epoch + 1, float(loss.detach().cpu()),
                        float(positive_score.detach().mean().cpu()),
                        float(negative_score.detach().mean().cpu())))
    model.eval()
    with torch.no_grad():
        embedding = model.encode(x, a).cpu().numpy()
    with (Path(folder) / f'dgi_loss_seed{seed}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'positive_score_mean', 'negative_score_mean'])
        writer.writerows(history)
    torch.save(model.state_dict(), Path(folder) / f'dgi_weights_seed{seed}.pt')
    return embedding, device
