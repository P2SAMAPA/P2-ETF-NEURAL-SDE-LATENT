import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-neural-sde-latent-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016, 4032]

# Latent SDE parameters
SEQ_LEN = 10                # number of past days as input
LATENT_DIM = 16
HIDDEN_DIM = 32
DT = 0.01
STEPS = 10
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

TOP_N = 3
