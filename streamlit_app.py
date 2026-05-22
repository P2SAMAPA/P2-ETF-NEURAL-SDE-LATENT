import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Neural SDE Latent", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌀 Neural SDE Latent Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Latent Stochastic Differential Equations | Variational inference | Next‑day prediction | Multi‑window evaluation (incl. 4032d)</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🌀 Latent SDE")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Latent dim:** {config.LATENT_DIM}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    # Corrected prefix to match the trainer output
    json_files = [f for f in files if f.endswith('.json') and 'latent_sde_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Latent SDE Predicted Return")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - The **Latent SDE** model learns a stochastic differential equation in a low‑dimensional latent space.
    - A recognition network (encoder) maps the observed return sequence to an initial latent distribution.
    - The latent state evolves according to a neural drift and diffusion (SDE), simulated via Euler‑Maruyama.
    - The final latent state is decoded to predict the next day's return.
    - The model is trained using variational inference (ELBO) with a KL regularisation term.
    - The predicted return is the score. Higher predicted return → stronger long signal.
    - For each ETF, the rolling window that gives the **highest predicted return** is selected.
    - New window: **4032 days** (approx 16 trading years) captures very long‑term dynamics.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">pred return = {etf['sde_pred']:.6f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Latent SDE Prediction": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Latent SDE Prediction"] = pd.to_numeric(df["Latent SDE Prediction"], errors='coerce')
            df = df.dropna(subset=["Latent SDE Prediction"]).sort_values("Latent SDE Prediction", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Latent SDE learns a stochastic latent dynamics model via amortised variational inference. Higher predicted return → stronger long signal.")
