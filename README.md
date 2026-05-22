# Neural SDE Latent Engine

Implements a latent stochastic differential equation (SDE) model (Li et al., 2020) for ETF return prediction. The model learns a latent dynamics SDE with neural drift and diffusion functions, inferred via amortised variational inference (VAE). The encoder maps observed return sequences to an initial latent distribution; the SDE is simulated forward; the decoder predicts the next day's return.

- **Latent SDE:** Drift & diffusion neural networks, Euler‑Maruyama integration
- **Inference:** Variational autoencoder (ELBO with KL regularisation)
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
