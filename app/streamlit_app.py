"""Run:  streamlit run app/streamlit_app.py   (after `python run_all.py`)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import config as C
from src.predict import load_reco_artifacts, load_yield_artifacts, predict_yield, recommend_crops

st.set_page_config(page_title="Indian Agriculture AI", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield & Recommendation")
st.caption("Ensemble regression + machine learning on Indian agriculture data")

if not (C.MODELS / "yield_best.joblib").exists():
    st.error("Models not found. Run `python run_all.py` first.")
    st.stop()


@st.cache_resource
def _load():
    return load_yield_artifacts(), load_reco_artifacts()


(y_model, y_meta), r_bundle = _load()
tab1, tab2 = st.tabs(["📈 Predict yield", "🌱 Recommend crop"])

with tab1:
    state = st.selectbox("State", sorted(y_meta["geo"]))
    district = st.selectbox("District", y_meta["geo"][state])
    crop = st.selectbox("Crop", y_meta["state_crops"][state])
    season = st.selectbox("Season", y_meta["crop_seasons"][crop])
    year = st.number_input("Year", y_meta["year_min"], 2035, y_meta["year_max"])
    area = st.number_input("Area (hectares)", 1.0, 1_000_000.0, float(y_meta["median_area"][crop]))
    rain = None
    if y_meta["has_rain"]:
        rain = st.number_input("Seasonal/annual rainfall (mm)", 0.0, 6000.0, float(y_meta["state_rain"][state]))
    if st.button("Predict yield", type="primary"):
        yld, prod = predict_yield(y_model, y_meta, state, district, crop, season, year, area, rain)
        c1, c2 = st.columns(2)
        c1.metric("Predicted yield", f"{yld:,.2f} t/ha")
        c2.metric("Implied production", f"{prod:,.0f} t")
        st.caption(f"Model: {y_meta['best_model']} · years beyond {y_meta['year_max']} are extrapolations.")

with tab2:
    rg = r_bundle["ranges"]
    cols = st.columns(2)
    vals = {}
    spec = [("N", "Nitrogen (N)"), ("P", "Phosphorus (P)"), ("K", "Potassium (K)"),
            ("temperature", "Temperature (°C)"), ("humidity", "Humidity (%)"),
            ("ph", "Soil pH"), ("rainfall", "Rainfall (mm)")]
    for i, (k, label) in enumerate(spec):
        lo, hi, med = rg[k]
        vals[k] = cols[i % 2].slider(label, float(lo), float(hi), float(med))
    if st.button("Recommend", type="primary"):
        res = recommend_crops(r_bundle, vals["N"], vals["P"], vals["K"], vals["temperature"],
                              vals["humidity"], vals["ph"], vals["rainfall"], top_k=3)
        for rank, (name, p) in enumerate(res, 1):
            st.write(f"**{rank}. {name.title()}** — {p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
