import streamlit as st
import joblib
import pandas as pd
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🎗️",
    layout="centered"
)

# ===================== STYLE =====================
st.markdown("""
<style>
.main {
    background-color: #FFF5F8;
}

section[data-testid="stSidebar"] {
    background-color: #FCE4EC;
}

.stButton > button {
    background-color: #E91E63;
    color: white;
    border-radius: 12px;
    font-size: 18px;
    height: 3em;
}

.stButton > button:hover {
    background-color: #D81B60;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

h1, h2, h3 {
    color: #C2185B;
}
</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline.pkl")

model = load_model()

# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>🎗️ Breast Cancer Risk Prediction</h1>", unsafe_allow_html=True)
st.write("Predict breast cancer risk using clinical and demographic features.")
st.divider()

# ================= SIDEBAR =================
st.sidebar.header("🧪 Patient Information")

year = st.sidebar.selectbox("Year", [2013])
age_group = st.sidebar.slider("Age Group (5-year intervals)", 1, 13, 7)

race_eth = st.sidebar.selectbox(
    "Race / Ethnicity",
    [1, 2, 3, 4, 5, 6]
)

first_degree_hx = st.sidebar.selectbox(
    "First Degree Family History",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

biophx = st.sidebar.selectbox(
    "Biopsy History",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ---------- Reproductive ----------
age_menarche_label = st.sidebar.selectbox(
    "Age at Menarche",
    ["Late", "Early", "Normal"]
)
age_menarche_map = {"Late": 0, "Early": 1, "Normal": 2}
age_menarche = age_menarche_map[age_menarche_label]

age_first_birth_label = st.sidebar.selectbox(
    "Age at First Birth",
    ["Early", "Normal", "Late", "Very Late"]
)
age_first_birth_map = {
    "Early": 1,
    "Normal": 2,
    "Late": 3,
    "Very Late": 4
}
age_first_birth = age_first_birth_map[age_first_birth_label]

menopaus_label = st.sidebar.selectbox(
    "Menopause Status",
    ["Pre", "Post", "Peri"]
)
menopaus_map = {"Pre": 1, "Post": 2, "Peri": 3}
menopaus = menopaus_map[menopaus_label]

# ---------- Medical ----------
birads_density = st.sidebar.selectbox(
    "BIRADS Breast Density",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Fatty",
        2: "Scattered",
        3: "Heterogeneously Dense",
        4: "Extremely Dense"
    }[x]
)

current_hrt = st.sidebar.selectbox(
    "Current HRT",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

bmi_group = st.sidebar.selectbox(
    "BMI Group",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Underweight",
        2: "Normal",
        3: "Overweight",
        4: "Obese"
    }[x]
)

# ================= INPUT DATAFRAME =================
input_df = pd.DataFrame({
    "year": [year],
    "age_group_5_years": [age_group],
    "race_eth": [race_eth],
    "first_degree_hx": [first_degree_hx],
    "age_menarche": [age_menarche],
    "age_first_birth": [age_first_birth],
    "BIRADS_breast_density": [birads_density],
    "current_hrt": [current_hrt],
    "menopaus": [menopaus],
    "bmi_group": [bmi_group],
    "biophx": [biophx]
})

st.subheader("📥 Input Data")
st.dataframe(input_df, use_container_width=True)

# ================= PREDICTION =================
if st.button("🔍 Predict Risk", use_container_width=True):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error("⚠️ High Risk of Breast Cancer")
    else:
        st.success("✅ Low Risk of Breast Cancer")

    st.metric("Low Risk Probability", f"{prob[0]*100:.2f}%")
    st.metric("High Risk Probability", f"{prob[1]*100:.2f}%")


# ===================== CSV UPLOAD =====================
st.sidebar.divider()
st.sidebar.header("📂 Batch Prediction (CSV)")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        csv_df = pd.read_csv(uploaded_file)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📄 Uploaded CSV")
        st.dataframe(csv_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("📊 Predict CSV", use_container_width=True):
            with st.spinner("🔄 Predicting CSV data..."):
                preds = model.predict(csv_df)
                probs = model.predict_proba(csv_df)

            csv_df["prediction"] = preds
            csv_df["risk_label"] = csv_df["prediction"].map(
                {0: "Low Risk", 1: "High Risk"}
            )
            csv_df["confidence"] = probs.max(axis=1)

            csv_df.to_csv("unknown_predictions.csv", index=False)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧠 Prediction Results")
            st.dataframe(csv_df, use_container_width=True)

            st.download_button(
                "⬇️ Download unknown_predictions.csv",
                data=csv_df.to_csv(index=False),
                file_name="unknown_predictions.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ CSV Error: {e}")

# ===================== FOOTER =====================
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Breast Cancer ML Project 🎓</p>",
    unsafe_allow_html=True
)
