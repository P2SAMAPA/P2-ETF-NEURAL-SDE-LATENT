import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class LatentSDE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32, dt=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        # Encoder: returns -> latent mean, logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        # Drift function (neural net)
        self.drift = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Diffusion function (neural net) – output positive diagonal
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder: latent -> predicted return
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        # x: (batch, input_dim)
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def sde_step(self, z, dt):
        # Euler–Maruyama step for latent SDE
        drift_val = self.drift(z)
        diff_val = torch.exp(self.diffusion(z))  # ensure positivity
        dw = torch.randn_like(z) * np.sqrt(dt)
        z_next = z + drift_val * dt + diff_val * dw
        return z_next

    def forward(self, x, steps=10):
        # Encode initial latent
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        # Simulate SDE
        for _ in range(steps):
            z = self.sde_step(z, self.dt)
        # Decode to prediction
        pred = self.decoder(z)
        return pred, mean, logvar

def train_latent_sde(X_train, y_train, input_dim, latent_dim=16, hidden_dim=32,
                     dt=0.01, steps=10, lr=1e-3, epochs=50, batch_size=32, device='cpu'):
    model = LatentSDE(input_dim, latent_dim, hidden_dim, dt).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    n = len(X_t)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            Xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            pred, mean, logvar = model(Xb, steps=steps)
            # Squeeze pred to match yb shape (remove extra dim if any)
            pred = pred.squeeze()
            # Reconstruction loss (MSE)
            recon_loss = nn.MSELoss()(pred, yb)
            # KL divergence for latent (standard Gaussian prior)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / len(Xb)
            loss = recon_loss + 0.01 * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def predict_latent_sde(model, X, steps=10):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        pred, _, _ = model(X_t, steps=steps)
    return pred.squeeze().cpu().numpy()
