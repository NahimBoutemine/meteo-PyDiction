import pandas as pd
import streamlit as st

st.title("PyDiction")
st.title("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
st.title("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")


uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
