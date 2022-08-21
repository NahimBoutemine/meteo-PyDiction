import pandas as pd
import streamlit as st

st.title("PyDiction")
st.header("PyDiction")

st.markdown("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
st.markdown("C'est un travail autour de la météorologie et du Machine Learning (ML).")
st.markdown("Procédure : il vous faudra charger des données de météorologie (températures etc) les plus conséquentes possibles ")
st.markdown("au FORMAT CSV. Ensuite, vous pourrez suivre les étapes et répondre aux questions jusqu'à la création du modèle et la prédiction de la présence ou non de pluie à J+1, sur un point quelconque du territoire australien. ")
st.markdown("Le score de prédiction sera affiché en fin. Les données rentrées devront être du même type que ce sur ce lien https://www.kaggle.com/jsphyg/weather-dataset-rattle-package, car le modèle est construit à partir de ces dernières")
st.markdown("C'est-à-dire sur 49 stations, sur plusieurs années, et comprenant les données complètes chaque jour jusqu'à la veille de la journée à prédire : ensoleillement, humidité, vitesse et sens du vent, quantité de nuages, températures minimales et maximales etc")
st.markdown("Les formats des données doivent être identiques notamment pour la date et la variable cible, RainTomorrow. Toute donnée supplémentaire peut être intéressante mais peut demander un traitement de conversion.")
st.markdown("La pluie est considérée comme présente si elle est strictement supérieure à 1mm. ")
st.markdown("En plus du guidage sur la préparation de vos données et la création du modèle, vous profiterez d'affichages de graphiques décrivant vos données ")
st.markdown("Vous pouvez donc commencer par charger vos données et observer leur prmeemière description! ")

st.header("Première partie")
uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)
  st.markdown("S'il existe des valeurs manquantes, elles sont à enlever pour le bon déroulement de la modélisation, de même que les doublons. ")
  st.markdown("Affichons ces fameuses valeurs manquantes :")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  st.write(percent_missing_df)
  st.markdown("Ainsi, Les valeurs manquantes sont enlevés automatiquement car, même si en général elles sont remplacées par une valeur (imputation statistique), selon notre expérience dans ce cas de prédiction cela ne fait que rajouter du temps de calcul")
  df = df.dropna()
  df = df = df.drop_duplicates()
  st.markdown("Vérifions à présent le nombre de données manquantes : on peut voir à présent qu'il n'y en a plus, ni de doublons!")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  st.write(percent_missing_df)
  
