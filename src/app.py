from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "random_forest_pipeline.joblib"

st.set_page_config(
    page_title="Online Shopping Purchase Predictor",
    page_icon="🛍️",
    layout="centered",
)

st.title("Online Shopping Purchase Predictor")
st.caption(
    "Predict whether an online shopping session is likely to result in a purchase."
)

if not MODEL_PATH.exists():
    st.error(
        "Saved model not found. Run the final notebook first so "
        "`artifacts/random_forest_pipeline.joblib` is created."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("shopping_predictor"):
    st.subheader("Browsing Activity")

    product_pages = st.number_input(
        "Product pages visited",
        min_value=0,
        value=10,
        step=1,
    )

    product_duration = st.number_input(
        "Time on product pages (seconds)",
        min_value=0.0,
        value=300.0,
        step=30.0,
    )

    bounce_rate = st.number_input(
        "Bounce rate",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01,
        format="%.4f",
    )

    exit_rate = st.number_input(
        "Exit rate",
        min_value=0.0,
        max_value=1.0,
        value=0.04,
        step=0.01,
        format="%.4f",
    )

    page_value = st.number_input(
        "Page value",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.2f",
    )

    visitor_type = st.selectbox(
        "Visitor type",
        ["Returning_Visitor", "New_Visitor", "Other"],
    )

    month = st.selectbox(
        "Month",
        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        index=8,
    )

    weekend = st.checkbox("Weekend session")

    submitted = st.form_submit_button(
        "Predict Purchase",
        use_container_width=True,
    )

if submitted:
    sample = pd.DataFrame([{
        "Administrative": 1,
        "Administrative_Duration": 30.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_value,
        "SpecialDay": 0.0,
        "Month": month,
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }])

    prediction = int(model.predict(sample)[0])
    purchase_probability = float(model.predict_proba(sample)[0, 1])

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:
        st.success("Purchase")
    else:
        st.info("No Purchase")

    st.metric(
        "Estimated purchase probability",
        f"{purchase_probability:.1%}",
    )

    st.caption(
        "Technical session-code fields not shown in the form use fixed demo values."
    )