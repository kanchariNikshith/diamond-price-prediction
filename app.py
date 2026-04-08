import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("diamond_knn_model.pkl", "rb"))

st.title("💎 Diamond Price Prediction")

st.write("Enter diamond details:")

# Inputs
carat = st.number_input("Carat", min_value=0.0, step=0.1)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth", min_value=0.0)
table = st.number_input("Table", min_value=0.0)
x = st.number_input("Length (x)", min_value=0.0)
y = st.number_input("Width (y)", min_value=0.0)
z = st.number_input("Height (z)", min_value=0.0)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: ${prediction[0]:.2f}")