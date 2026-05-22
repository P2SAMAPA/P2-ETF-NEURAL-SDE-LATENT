import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from neural_sde_latent import train_latent_sde, predict_latent_sde

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def create_sequences(returns_series, seq_len=10):
    """Create sliding windows of returns."""
    if len(returns_series) < seq_len + 1:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(returns_series)-1):
        X.append(returns_series.iloc[i-seq_len:i].values)
        y.append(returns_series.iloc[i+1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # Reshape X to (batch, seq_len, 1) for the model
    X = X.reshape(-1, seq_len, 1)
    return X, y

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Neural SDE Latent) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + config.SEQ_LEN + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + config.SEQ_LEN + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:]
            etf_scores = {}
            for etf in tickers:
                if etf not in ret_win.columns:
                    continue
                series = ret_win[etf].dropna()
                X, y = create_sequences(series, seq_len=config.SEQ_LEN)
                if X is None or len(X) < 10:
                    print(f"    {etf}: not enough sequences")
                    continue
                split = int(0.8 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                model = train_latent_sde(X_train, y_train,
                                         latent_dim=config.LATENT_DIM,
                                         hidden_dim=config.HIDDEN_DIM,
                                         dt=config.DT,
                                         steps=config.STEPS,
                                         lr=config.LEARNING_RATE,
                                         epochs=config.EPOCHS,
                                         batch_size=config.BATCH_SIZE,
                                         device=device)
                last_X = X[-1:].reshape(1, config.SEQ_LEN, 1)
                pred = predict_latent_sde(model, last_X)[0]   # <-- FIXED: [0] not [0,0]
                etf_scores[etf] = pred
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "sde_pred": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by latent SDE prediction: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/latent_sde_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Neural SDE Latent Engine complete ===")

if __name__ == "__main__":
    main()
