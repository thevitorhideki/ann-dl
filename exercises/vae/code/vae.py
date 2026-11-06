"""VAE para MNIST, com reparametrização e KL escritas explicitamente."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


# --8<-- [start:data]
def load_data(data_dir):
    train_full = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(
        data_dir, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(
        train_full, [50_000, 10_000],
        generator=torch.Generator().manual_seed(42))

    def flatten_once(dataset):
        # Aplico ToTensor uma vez e guardo os vetores para não repetir a
        # conversão de imagens PIL em todas as 20 épocas. A ordem é mantida.
        images, labels = [], []
        for x, y in DataLoader(dataset, batch_size=1024, shuffle=False):
            images.append(x.flatten(start_dim=1))
            labels.append(y)
        return TensorDataset(torch.cat(images), torch.cat(labels))

    return tuple(flatten_once(part) for part in (train, val, test))
# --8<-- [end:data]


# --8<-- [start:model]
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        # logvar = log(sigma²), então sigma = exp(logvar / 2).
        # O ruído independe dos pesos e o gradiente passa por mu e sigma.
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(reconstruction, x, mu, logvar):
    # Somo pixels na BCE e dimensões latentes na KL. Divido só pelo
    # número de imagens, mantendo os dois termos em nats por imagem.
    batch_size = x.shape[0]
    bce = F.binary_cross_entropy(reconstruction, x, reduction="sum") / batch_size
    kl = 0.5 * torch.sum(mu.square() + logvar.exp() - 1 - logvar) / batch_size
    return bce, kl
# --8<-- [end:model]


# --8<-- [start:training]
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_bce, total_kl, n_images = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            if training:
                optimizer.zero_grad()
            reconstruction, mu, logvar = model(x)
            bce, kl = loss_terms(reconstruction, x, mu, logvar)
            if training:
                (bce + kl).backward()  # beta = 1 em todos os experimentos.
                optimizer.step()
            # Ponderar pelo tamanho evita dar peso excessivo ao último batch.
            n = x.shape[0]
            total_bce += bce.item() * n
            total_kl += kl.item() * n
            n_images += n
    return {"reconstruction": total_bce / n_images,
            "kl": total_kl / n_images,
            "negative_elbo": (total_bce + total_kl) / n_images}


def evaluate(model, dataset):
    # Uma amostra de q(z|x) por imagem estima a esperança da reconstrução.
    # Reutilizo o ruído na avaliação sem alterar o RNG do treinamento.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        return run_epoch(model, loader)


def train_model(latent_dim, train, val):
    torch.manual_seed(42)  # Antes de construir CADA modelo.
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train, batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    history = []
    for epoch in range(1, 21):
        train_metrics = run_epoch(model, loader, optimizer)
        val_metrics = evaluate(model, val)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(f"d={latent_dim:2d}, época {epoch:2d}: "
              f"treino BCE={train_metrics['reconstruction']:.3f}, "
              f"KL={train_metrics['kl']:.3f}; "
              f"val BCE={val_metrics['reconstruction']:.3f}, "
              f"KL={val_metrics['kl']:.3f}", flush=True)
    # Orçamento fixo: uso a época 20, sem escolher pelo teste.
    model.eval()
    return model, history
# --8<-- [end:training]
