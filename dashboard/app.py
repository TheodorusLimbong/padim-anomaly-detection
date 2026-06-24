import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.utils import (
    list_experiments, load_model, load_experiment_data,
    preprocess_and_extract, infer_padim, infer_knn, normalize_map,
)
from PIL import Image

st.set_page_config(page_title="PaDiM vs KNN — Live", layout="centered")
st.title("🔬 PaDiM vs KNN — Live Anomaly Detection")
st.caption("MVTec AD Bottle · ResNet-50 + MoCo v2 · PaDiM vs KNN")

experiments = list_experiments()
if not experiments:
    st.error("Tidak ada experiment di output/experiments/. Jalankan `python experiments/run_padim.py` dulu.")
    st.stop()

exp_name = st.sidebar.selectbox("Experiment:", [e[0] for e in experiments], index=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_model(_dev):
    return load_model(_dev)

@st.cache_resource
def get_exp_data(_name, _dev):
    return load_experiment_data(_name, _dev)

model, extractor = get_model(device)
exp_data = get_exp_data(exp_name, device)

uploaded = st.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    with st.spinner("Feature extraction..."):
        patches = preprocess_and_extract(pil_image, extractor, exp_data, device)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(pil_image, caption="Original", use_container_width=True)

    with col2:
        with st.spinner("PaDiM inference..."):
            padim = infer_padim(patches, exp_data, device)

        map_raw = padim["map_raw"]
        pmin = exp_data.get("padim_map_min")
        pmax = exp_data.get("padim_map_max")
        if pmax is not None and pmax > pmin:
            map_norm = (map_raw - pmin) / (pmax - pmin)
        else:
            map_norm = normalize_map(map_raw)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.clip(map_norm, 0, 1), cmap="jet", vmin=0, vmax=1)
        s = padim["score_norm"] if padim["score_norm"] is not None else padim["score"]
        ax.set_title(f"PaDiM\n{s:.4f}", fontsize=11)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        pred = padim["is_anomaly"]
        if pred is True:
            st.markdown('<p style="background:#d32f2f;color:white;text-align:center;padding:4px;border-radius:4px;font-weight:bold">🔴 ANOMALY</p>', unsafe_allow_html=True)
        elif pred is False:
            st.markdown('<p style="background:#388e3c;color:white;text-align:center;padding:4px;border-radius:4px;font-weight:bold">✅ NORMAL</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="background:#757575;color:white;text-align:center;padding:4px;border-radius:4px">Score: {padim["score"]:.4f}</p>', unsafe_allow_html=True)
        st.caption(f"⏱ {padim['time']:.3f}s")

    with col3:
        with st.spinner("KNN inference..."):
            knn = infer_knn(patches, exp_data, device)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(normalize_map(knn["map_raw"]), cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"KNN\n{knn['score']:.4f}", fontsize=11)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        pred = knn["is_anomaly"]
        if pred is True:
            st.markdown('<p style="background:#d32f2f;color:white;text-align:center;padding:4px;border-radius:4px;font-weight:bold">🔴 ANOMALY</p>', unsafe_allow_html=True)
        elif pred is False:
            st.markdown('<p style="background:#388e3c;color:white;text-align:center;padding:4px;border-radius:4px;font-weight:bold">✅ NORMAL</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="background:#757575;color:white;text-align:center;padding:4px;border-radius:4px">Score: {knn["score"]:.4f}</p>', unsafe_allow_html=True)
        st.caption(f"⏱ {knn['time']:.3f}s")

    st.divider()
    m = exp_data["metrics"]
    cols = st.columns(4)
    cols[0].metric("PaDiM AUROC", f'{m["padim"]["auroc"]:.4f}')
    cols[1].metric("PaDiM PRO", f'{m["padim"]["pro_score"]:.4f}')
    cols[2].metric("KNN AUROC", f'{m["knn"]["auroc"]:.4f}')
    cols[3].metric("KNN PRO", f'{m["knn"]["pro_score"]:.4f}')

    with st.expander("Detail"):
        st.json({
            "experiment": exp_name,
            "n_train": m["padim"]["n_train"],
            "PaDiM threshold": exp_data["threshold_padim"],
            "KNN threshold": exp_data["threshold_knn"],
            "PaDiM time (s)": round(padim["time"], 3),
            "KNN time (s)": round(knn["time"], 3),
        })
else:
    st.info("Upload gambar untuk deteksi anomali.")
