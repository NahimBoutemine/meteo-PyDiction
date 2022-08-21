import pandas as pd
import streamlit as st

st.title("PyDiction")
st.mark("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
st.mark("C'est un travail autour de la météorologie et du Machine Learning (ML).")
st.mark("Procédure : il vous faudra charger des données de météorologie (températures etc) les plus conséquentes possibles ")
st.mark("au FORMAT CSV. Ensuite, vous pourrez suivre les étapes et répondre aux questions jusqu'à la prédiction de la présence de pluie ")
st.mark("à J+1, sur un point quelconque du territoire australien.")
st.mark("Le score de prédiction sera affiché en fin. Les données rentrées devront être du type ")

st.title("Première partie")


uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)

