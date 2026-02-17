import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model
model = pickle.load(open("model.pkl","rb"))

# ---------- Custom CSS ----------
st.markdown("""
<style>
.big-font {
    font-size:35px !important;
    font-weight:700;
    color:#2E86C1;
}
.card {
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
    background-color:#f9f9f9;
}
.footer {
    text-align:center;
    color:gray;
    padding:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="big-font">🏠 AI House Price Prediction System</p>', unsafe_allow_html=True)
st.write("Predict property prices using Machine Learning")

st.divider()

# ---------- Layout ----------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Enter House Details")

    MSSubClass = st.number_input("MSSubClass",20,200,60)
    MSZoning = st.selectbox("MSZoning",["RL","RM","FV","RH","C (all)"])
    LotArea = st.number_input("Lot Area",1000,20000,5000)
    LotConfig = st.selectbox("LotConfig",["Inside","Corner","CulDSac","FR2","FR3"])
    BldgType = st.selectbox("BldgType",["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"])

with col2:
    st.subheader("Property Features")

    OverallCond = st.slider("Overall Condition",1,10,5)
    YearBuilt = st.number_input("Year Built",1900,2024,2000)
    YearRemodAdd = st.number_input("Remodel Year",1900,2024,2005)
    Exterior1st = st.selectbox("Exterior",["VinylSd","MetalSd","Wd Sdng","HdBoard","BrkFace"])
    BsmtFinSF2 = st.number_input("Basement Finished SF",0,2000,0)
    TotalBsmtSF = st.number_input("Total Basement SF",0,3000,800)

st.divider()

# ---------- Predict Button ----------
if st.button("🚀 Predict House Price"):

    HouseAge = 2024 - YearBuilt
    RemodelAge = 2024 - YearRemodAdd
    TotalArea = LotArea + TotalBsmtSF

    data = pd.DataFrame([[MSSubClass,MSZoning,LotArea,LotConfig,BldgType,
                          OverallCond,YearBuilt,YearRemodAdd,Exterior1st,
                          BsmtFinSF2,TotalBsmtSF,
                          HouseAge,RemodelAge,TotalArea]],
                        columns=["MSSubClass","MSZoning","LotArea","LotConfig","BldgType",
                                 "OverallCond","YearBuilt","YearRemodAdd","Exterior1st",
                                 "BsmtFinSF2","TotalBsmtSF",
                                 "HouseAge","RemodelAge","TotalArea"])

    prediction = np.expm1(model.predict(data)[0])

    st.divider()
    st.subheader("📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    c1.metric("Estimated Price", f"${prediction:,.0f}")
    c2.metric("House Age", f"{HouseAge} yrs")
    c3.metric("Total Area", f"{TotalArea} sqft")

    if prediction > 300000:
        st.success("🏡 Luxury Property")
    elif prediction > 150000:
        st.info("🏠 Mid-range Property")
    else:
        st.warning("🏚️ Budget Property")

# ---------- Footer ----------
st.markdown('<div class="footer">Made with ❤️ using Machine Learning & Streamlit</div>', unsafe_allow_html=True)