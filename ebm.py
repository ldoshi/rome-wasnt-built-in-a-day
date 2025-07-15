import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import matplotlib.pyplot as plt


# --- Bars and Stripes Dataset (4x4) ---
def generate_bars_stripes(n=4):
    images = []
    for row_pattern in itertools.product([0, 1], repeat=n):
        image = np.tile(np.array(row_pattern).reshape(n, 1), (1, n))
        images.append(image)
    for col_pattern in itertools.product([0, 1], repeat=n):
        image = np.tile(np.array(col_pattern).reshape(1, n), (n, 1))
        images.append(image)
    # Remove duplicates
    unique = []
    for img in images:
        if not any(np.array_equal(img, u) for u in unique):
            unique.append(img)
    return np.array(unique).astype(np.float32)


data_np = generate_bars_stripes(4)
data = torch.tensor(data_np.reshape(len(data_np), -1))  # shape: (N, 16)


# --- EBM: MLP Energy Model ---
class EBM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,)


model = EBM(16)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --- Sampling (Gibbs-style) ---
@torch.no_grad()
def gibbs_sample(model, x_init, steps=30):
    x = x_init.clone()
    for _ in range(steps):
        for i in range(x.shape[1]):
            x_flip = x.clone()
            x_flip[:, i] = 1 - x_flip[:, i]  # Flip bit i
            e_orig = model(x)
            e_flip = model(x_flip)
            prob = torch.sigmoid(e_orig - e_flip)  # Lower energy = more likely
            mask = (torch.rand(x.size(0)) < prob).float()
            x[:, i] = x[:, i] * mask + x_flip[:, i] * (1 - mask)
    return x


# --- Training Loop ---
epochs = 1000
batch_size = 64
for epoch in range(epochs):
    idx = torch.randint(0, data.size(0), (batch_size,))
    x_data = data[idx]

    x_noise = torch.bernoulli(torch.full_like(x_data, 0.5))
    x_neg = gibbs_sample(model, x_noise, steps=40)

    energy_pos = model(x_data)
    energy_neg = model(x_neg)

    loss = (energy_pos - energy_neg).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# --- Visualize Generated Samples ---
samples = gibbs_sample(model, torch.bernoulli(torch.full_like(data, 0.5)), steps=100)
samples = samples[:16].reshape(-1, 4, 4)

fig, axs = plt.subplots(4, 4, figsize=(6, 6))
for ax, img in zip(axs.flat, samples):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.show()
