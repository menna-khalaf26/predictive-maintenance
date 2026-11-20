import os
import joblib, pandas as pd, streamlit as st
from features import add_features, ALL_NUM, CAT

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧")

@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    try:
        return joblib.load("model.joblib")
    except Exception:            # e.g. scikit-learn version mismatch -> retrain once
        import train; train.main()
        return joblib.load("model.joblib")

bundle = load_bundle()
pipe, THR, M = bundle["pipeline"], bundle["threshold"], bundle["metrics"]

st.title("🔧 Predictive Maintenance (AI4I 2020)")
st.caption(f"Model: {M['model']} · test recall {M['recall']} · precision {M['precision']} · PR-AUC {M['pr_auc']}")
st.write("Enter the machine's sensor readings to estimate failure risk.")

c1, c2 = st.columns(2)
with c1:
    mtype = st.radio("Product quality type", ["L", "M", "H"], index=1, horizontal=True)
    air = st.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    proc = st.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
with c2:
    rpm = st.slider("Rotational speed [rpm]", 1100, 3000, 1500)
    torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.1)
    wear = st.slider("Tool wear [min]", 0, 260, 100)

if st.button("Predict", type="primary"):
    df = add_features(pd.DataFrame([{"Type": mtype, "air_temp": air, "process_temp": proc,
                                     "rpm": rpm, "torque": torque, "tool_wear": wear}]))
    p = float(pipe.predict_proba(df[ALL_NUM + CAT])[0, 1])
    st.progress(min(p, 1.0))
    if p >= THR:
        st.error(f"⚠️ Maintenance recommended — failure probability {p:.1%} (alert threshold {THR:.1%})")
    else:
        st.success(f"✅ Operating normally — failure probability {p:.1%} (alert threshold {THR:.1%})")
