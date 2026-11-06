"""Reproduz os dois treinamentos, os resultados e as seis figuras do relatório.

Na raiz do repositório: python docs/exercises/vae/code/run_experiment.py
"""

# --8<-- [start:setup]
from pathlib import Path
import csv
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

if "__file__" in globals():
    BASE = Path(__file__).resolve().parents[1]
else:
    BASE = Path.cwd()
    if not (BASE / "code" / "vae.py").is_file():
        BASE = BASE / "docs" / "exercises" / "vae"
sys.path.insert(0, str(BASE / "code"))
from vae import load_data, VAE, loss_terms, run_epoch, evaluate, train_model

ROOT = BASE.parents[2]
FIGURES = BASE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
np.random.seed(42)
plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def save_history(history, filename):
    with (BASE / "code" / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
# --8<-- [end:setup]

# --8<-- [start:data]
train, val, test = load_data(ROOT / ".cache" / "vae" / "data")
print(f"Treino: {len(train)}; validação: {len(val)}; teste: {len(test)}")
print(f"Entrada: {train.tensors[0].shape}; pixels em "
      f"[{train.tensors[0].min().item()}, {train.tensors[0].max().item()}]")
# --8<-- [end:data]

# --8<-- [start:train16]
model16, history16 = train_model(16, train, val)
save_history(history16, "history-d16.csv")
test16 = evaluate(model16, test)
print("Teste d=16 (nats/imagem):", test16)
# --8<-- [end:train16]

# --8<-- [start:figure1]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, term, title in zip(axes, ["reconstruction", "kl"], ["Reconstrução (BCE)", "KL"]):
    for split, label in [("train", "Treino"), ("val", "Validação")]:
        ax.plot(range(1, 21), [h[f"{split}_{term}"] for h in history16], label=label)
    ax.set(title=title, xlabel="Época", ylabel="Nats por imagem", xticks=[1, 5, 10, 15, 20])
    ax.legend()
fig.suptitle("Figura 1 — Termos da perda do VAE com d = 16")
save_figure(fig, "fig01-perdas.png")
# --8<-- [end:figure1]

# --8<-- [start:figure2]
x_test, y_test = test.tensors
indices = [int(torch.where(y_test == digit)[0][0]) for digit in range(10)]
originals = x_test[indices]
with torch.no_grad():
    mu, _ = model16.encode(originals)
    reconstructions = model16.decode(mu).reshape(-1, 28, 28)

fig, axes = plt.subplots(2, 10, figsize=(14, 3.7))
for col, (digit, index) in enumerate(zip(range(10), indices)):
    for row, image in enumerate([originals[col].reshape(28, 28), reconstructions[col]]):
        ax = axes[row, col]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set(xticks=[], yticks=[])
        if row == 0:
            ax.set_title(f"{digit} (índice {index})", fontsize=9)
        if row == 1:
            ax.set_xlabel("Coluna do pixel", fontsize=7)
    axes[0, col].set_ylabel("Linha do pixel", fontsize=7)
axes[0, 0].set_ylabel("Original\nLinha do pixel")
axes[1, 0].set_ylabel("Reconstrução\nLinha do pixel")
fig.suptitle("Figura 2 — Primeira imagem de cada dígito e reconstrução de μ (d = 16)")
save_figure(fig, "fig02-reconstrucoes.png")
print("Primeiros índices de teste por dígito 0–9:", indices)
# --8<-- [end:figure2]

# --8<-- [start:figure3]
torch.manual_seed(0)
with torch.no_grad():
    # Amostragem do prior escrita diretamente, sem helper de VAE.
    z = torch.randn(64, 16)
    samples = model16.decode(z).reshape(8, 8, 28, 28)
canvas = samples.permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
ax.set(title="Figura 3 — 64 amostras do prior N(0, I), d = 16, seed 0",
       xlabel="Coluna da amostra", ylabel="Linha da amostra",
       xticks=np.arange(8) * 28 + 13.5, xticklabels=np.arange(1, 9),
       yticks=np.arange(8) * 28 + 13.5, yticklabels=np.arange(1, 9))
save_figure(fig, "fig03-amostras.png")
# --8<-- [end:figure3]

# --8<-- [start:train2]
model2, history2 = train_model(2, train, val)
save_history(history2, "history-d2.csv")
test2 = evaluate(model2, test)
print("Teste d=2 (nats/imagem):", test2)
print("Comparação: d, reconstrução, KL, ELBO negativo")
for d, metrics in [(16, test16), (2, test2)]:
    print(d, *(round(metrics[k], 6) for k in ["reconstruction", "kl", "negative_elbo"]))
# --8<-- [end:train2]

# --8<-- [start:figure4]
with torch.no_grad():
    # Não há nova amostragem: o scatter usa a média codificada de cada imagem.
    encoded = torch.cat([model2.encode(x)[0] for x in x_test.split(128)]).numpy()
fig, ax = plt.subplots(figsize=(8.5, 7))
colors = plt.get_cmap("tab10")
for digit in range(10):
    points = encoded[y_test.numpy() == digit]
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.45,
               color=colors(digit), label=str(digit), rasterized=True)
ax.set(title="Figura 4 — Médias latentes das 10.000 imagens de teste (d = 2)",
       xlabel="μ₁", ylabel="μ₂", aspect="equal")
ax.legend(title="Dígito", bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
save_figure(fig, "fig04-espaco-latente.png")
# --8<-- [end:figure4]

# --8<-- [start:figure5]
quantiles = np.linspace(0.05, 0.95, 15)
coordinates = norm.ppf(quantiles)
# A primeira linha é o maior z2: a posição vertical coincide com o scatter.
latent_grid = np.array([(z1, z2) for z2 in coordinates[::-1] for z1 in coordinates])
with torch.no_grad():
    decoded_grid = model2.decode(torch.tensor(latent_grid, dtype=torch.float32))
tiles = decoded_grid.reshape(15, 15, 28, 28).permute(0, 2, 1, 3).reshape(420, 420)
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(tiles, cmap="gray", vmin=0, vmax=1)
positions = np.arange(15) * 28 + 13.5
ax.set(title="Figura 5 — Plano latente decodificado em 15 × 15 quantis (d = 2)",
       xlabel="z₁ = Φ⁻¹(q₁)", ylabel="z₂ = Φ⁻¹(q₂)",
       xticks=positions, xticklabels=[f"{z:.2f}" for z in coordinates],
       yticks=positions, yticklabels=[f"{z:.2f}" for z in coordinates[::-1]])
ax.tick_params(axis="x", labelrotation=90, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
save_figure(fig, "fig05-plano-decodificado.png")
# --8<-- [end:figure5]

# --8<-- [start:figure6]
index_a, index_b = 0, 1  # Primeiras duas imagens oficiais: dígitos 7 e 2.
t = torch.linspace(0, 1, 10).unsqueeze(1)
xa, xb = x_test[index_a:index_a + 1], x_test[index_b:index_b + 1]
pixel_interpolation = (1 - t) * xa + t * xb
with torch.no_grad():
    mua, _ = model16.encode(xa)
    mub, _ = model16.encode(xb)
    latent_interpolation = model16.decode((1 - t) * mua + t * mub)
fig, axes = plt.subplots(2, 10, figsize=(14, 3.7))
for row, images in enumerate([pixel_interpolation, latent_interpolation]):
    for col, image in enumerate(images):
        ax = axes[row, col]
        ax.imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set(xticks=[], yticks=[])
        if row == 0:
            ax.set_title(f"t = {t[col].item():.2f}")
        else:
            ax.set_xlabel("Coluna do pixel", fontsize=7)
axes[0, 0].set_ylabel("Pixels\nLinha do pixel")
axes[1, 0].set_ylabel("Latente (d = 16)\nLinha do pixel")
fig.suptitle("Figura 6 — Interpolação do índice 0 (7) ao índice 1 (2)")
save_figure(fig, "fig06-interpolacao.png")
print(f"Interpolação: índice {index_a} (dígito {y_test[index_a].item()}) → "
      f"índice {index_b} (dígito {y_test[index_b].item()})")
# --8<-- [end:figure6]

# --8<-- [start:results]
results = {
    "test_d16": test16, "test_d2": test2,
    "validation_kl_d16_first": history16[0]["val_kl"],
    "validation_kl_d16_last": history16[-1]["val_kl"],
    "first_test_indices_by_digit": indices,
    "interpolation_indices": [index_a, index_b],
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "numpy": np.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                    "device": "cpu", "threads": torch.get_num_threads(),
                    "evaluation_seed": 123, "evaluation_samples_per_image": 1}}
(BASE / "code" / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
# --8<-- [end:results]
