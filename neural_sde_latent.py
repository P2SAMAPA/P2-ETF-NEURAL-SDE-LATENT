import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DriftNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class DiffusionNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        # Softplus ensures positive diffusion
        return torch.nn.functional.softplus(self.net(z))

class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
    def forward(self, x):
        params = self.net(x)
        mean, logvar = params.chunk(2, dim=-1)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    def forward(self, z):
        return self.net(z)

class LatentSDE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim, drift_layers=2, diffusion_layers=2, dt=0.01, n_steps=20):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.drift = DriftNet(latent_dim, hidden_dim, drift_layers)
        self.diffusion = DiffusionNet(latent_dim, hidden_dim, diffusion_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, obs_dim)
        self.dt = dt
        self.n_steps = n_steps
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward_sde(self, z0, steps=None):
        if steps is None:
            steps = self.n_steps
        z = z0
        trajectory = [z]
        for t in range(steps):
            drift = self.drift(z)
            diffusion = self.diffusion(z)
            noise = torch.randn_like(z) * np.sqrt(self.dt)
            z = z + drift * self.dt + diffusion * noise
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)  # (batch, steps+1, latent_dim)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, obs_dim)
        # Encode the first observation to get initial latent
        mean, logvar = self.encoder(x_seq[:, 0, :])
        z0 = self.reparameterize(mean, logvar)
        # Simulate SDE forward
        z_path = self.forward_sde(z0, steps=self.n_steps)
        # Decode the final latent to predict next observation
        # We'll use the last time step of the latent path
        z_final = z_path[:, -1, :]
        pred = self.decoder(z_final)
        return pred, mean, logvar, z_path

def train_latent_sde(X_train, y_train, obs_dim, latent_dim=16, hidden_dim=64,
                     drift_layers=2, diffusion_layers=2, dt=0.01, n_steps=20,
                     lr=1e-3, epochs=50, batch_size=32, kl_weight=0.001, device='cpu'):
    model = LatentSDE(obs_dim, latent_dim, hidden_dim, drift_layers, diffusion_layers, dt, n_steps).to(device)
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
            pred, mean, logvar, _ = model.forward(Xb)
            recon_loss = nn.MSELoss()(pred, yb)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / Xb.size(0)
            loss = recon_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")
    return model

def predict_latent_sde(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        pred, _, _, _ = model.forward(X_t)
        return pred.cpu().numpy()
