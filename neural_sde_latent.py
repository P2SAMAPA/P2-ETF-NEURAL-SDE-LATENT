import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LatentSDE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32, dt=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        # Encoder: expects input of shape (batch, input_dim) where input_dim = seq_len * 1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.drift = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # output single value (next return)
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
        drift_val = self.drift(z)
        diff_val = torch.exp(self.diffusion(z))
        dw = torch.randn_like(z) * np.sqrt(dt)
        z_next = z + drift_val * dt + diff_val * dw
        return z_next

    def forward(self, x, steps=10):
        # x: (batch, seq_len, 1) -> flatten to (batch, seq_len)
        batch, seq_len, _ = x.shape
        x_flat = x.view(batch, -1)   # (batch, seq_len)
        mean, logvar = self.encode(x_flat)
        z = self.reparameterize(mean, logvar)
        for _ in range(steps):
            z = self.sde_step(z, self.dt)
        pred = self.decoder(z).squeeze(-1)   # (batch,)
        return pred, mean, logvar

def train_latent_sde(X_train, y_train, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 50)
    lr = kwargs.get('lr', 1e-3)
    latent_dim = kwargs.get('latent_dim', 16)
    hidden_dim = kwargs.get('hidden_dim', 32)
    dt = kwargs.get('dt', 0.01)
    steps = kwargs.get('steps', 10)
    # input_dim = seq_len (because we flatten)
    input_dim = X_train.shape[1] * X_train.shape[2]  # seq_len * 1 = seq_len
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
            recon_loss = nn.MSELoss()(pred, yb)
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
    return pred.cpu().numpy()
