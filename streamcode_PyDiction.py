import pandas as pd
import streamlit as st

st.title("Hello world!")

uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(dataframe)
