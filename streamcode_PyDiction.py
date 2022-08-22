
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error as mae
from sklearn import model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import metrics

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
st.markdown("En plus du guidage sur la préparation de vos données et la création du modèle, vous profiterez d'affichages de'indicateurs et de graphiques décrivant vos données ")
st.markdown("Vous pouvez donc commencer par charger vos données et observer leur première description! ")

st.header("Première partie")

#si le fichier est chargé, alors lancer le code seulement ensuite (condition nécessaire sinon le code se lance trop tôt et bloque):
uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  
  st.markdown("Affichons le nombre de lignes, vous devez en avoir au moins 56 000 après suppression des manquantes :")
  st.write(len(df))

  st.write(df)
  st.markdown("S'il existe des valeurs manquantes, elles sont à enlever pour le bon déroulement de la modélisation, de même que les doublons. ")
  st.markdown("Affichons le pourcentage de ces fameuses valeurs manquantes, et ce pour chacune des variables :")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  st.write(percent_missing_df)
  st.markdown("Ainsi, les valeurs manquantes sont enlevés automatiquement car, même si en général elles sont remplacées par une valeur (imputation statistique), selon notre expérience dans ce cas de prédiction cela ne fait que rajouter du temps de calcul")
  df = df.dropna()
  df = df.drop_duplicates()
  st.markdown("Vérifiez par vous-même à présent le nombre de données manquantes : on peut voir à présent qu'il n'y en a plus!")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  st.write(percent_missing_df)
  st.markdown("Affichons le pourcentage de valeurs non en doublon pour vérifier qu'elles ont été supprimées:")
  percentage_dupli = df.duplicated(keep=False).value_counts(normalize=True) * 100
  st.write(percentage_dupli)
  st.markdown("Affichons de nouveau le nombre de lignes, pour rappel vous devez en avoir au moins 56 000 à ce stade afin (selon notre expérience) d'avoir un score de prédiction suffisant, sinon vous devez compléter votre dataset :")
  st.write(len(df))
  st.markdown("A présent, les données sont automatiquement encodées, date est transformée en Année, Mois et Jours")
  
  #afficher la répartition des valeurs dans la cible A FAIRE ET DECRIRE
  
  #encodage des données
  
  ##Traitement de la variable 'date' :
  #Toutes les variables doivent être encodées et comme la pluie est un phénomène qui semble dépendant des saisons,
  # il serait interessant de décomposer Date. Pour cela, nous utilisons la méthode 
  #dt.datetime pour extraire année, mois, et jour
  df['year'] = pd.to_datetime(df['Date']).dt.year
  df['month'] = pd.to_datetime(df['Date']).dt.month
  df['day'] = pd.to_datetime(df['Date']).dt.day

  #réenregistrement des variables year, month, et day, en tant que int.
  df['year'] = df['year'].astype(int)
  df['month'] = df['month'].astype(int)
  df['day'] = df['day'].astype(int)
  #on élimine la colonne Date, désormais inutile et dont l'information a été conservée.
  df = df.drop('Date', axis = 1)
  ##Renommer pour lisibilité les booléénnes : 
  df['RainToday_encode'] = df['RainToday']
  df['RainTomorrow_encode'] = df['RainTomorrow']
  df = df.drop(labels = ['RainTomorrow', 'RainToday'], axis = 1)

  #import: 
  le = preprocessing.LabelEncoder()

  #encodage :     
  for var in df.select_dtypes(include='object').columns:
    df[var] = le.fit_transform(df[var])
  st.markdown("Voyez par vous-même les créations de variables et l'encodage :")
  st.write(df)

  #heatmap
  st.markdown("A présent, il faut sélectionner les variables explicatives pour la modélisation.")
  st.markdown("Pour cela, nous allons afficher la matrice des corrélations")
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)
  st.markdown("Suppression des variables explicatives corréllées à moins de 5% à la cible selon le test de Pearson qui sont 'WindDir3pm','Temp9am','WindDir9am'") 
  df = df.drop(['WindDir3pm','Temp9am','WindDir9am'], axis = 1)

  ##Découpage des données en jeu d'entrainement et jeu test pour
  #permettre d'évaluer la performance globale des modèles sur un jeu nouveau,
  #à hauteur de 20 % des données en test, avec en paramètre randomstate=42 rempli pour la reproductibilité 
  #des résultats :
  #(méthode courante d'évaluation):
  y = df['RainTomorrow_encode']
  x = df.drop('RainTomorrow_encode', axis = 1)

  #préparation de l'oversampling SMOTE
  smo = SMOTE()
  x_sm, y_sm = smo.fit_resample(x, y)
  st.write('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))

  # réduction du jeu de données, répartition en jeux train et test
  x_sm_train, x_sm_test, y_sm_train, y_sm_test = train_test_split(x_sm, y_sm, test_size=0.20, random_state=42)

  #itération du modèle : KNN, avec les données réduites.
  model = KNeighborsClassifier(metric='manhattan', n_neighbors=26, weights='distance') #mettre ici le meilleur nbr_voisins trouvé plus haut
  model.fit(x_sm_train,y_sm_train)

  ##Précision et f1-score :
  y_pred_train_KNNsm = model.predict(x_sm_train)
  y_pred_test_KNNsm = model.predict(x_sm_test)

  #accuracy : 
  acc_train_KNNsm  = accuracy_score(y_sm_train, y_pred_train_KNNsm)
  acc_test_KNNsm  = accuracy_score(y_sm_test, y_pred_test_KNNsm)
  st.write("acc_train : ", acc_train_KNNsm, "acc_test :", acc_test_KNNsm)

  ##F1-score :
  f1score_train_KNNsm = f1_score(y_sm_train, y_pred_train_KNNsm, average='macro')
  f1score_test_KNNsm = f1_score(y_sm_test, y_pred_test_KNNsm, average='macro')
  st.write("F1score_train : ", f1score_train_KNNsm, "F1score_test : ", f1score_test_KNNsm)

  #matrice de confusion : 
  st.write(pd.crosstab(y_sm_test, y_pred_test_KNNsm, rownames=['Classe réelle'], colnames=['Classe prédite']))

  #résultats :
  st.markdown("hausse à la fois de l'accuracy et du f1 score (0.87 et 0.87 contre 0.85 et 0.74 respectivement pour le KNN sans oversampling")
  st.markdown("meilleur classement des positifs !")

  #AUC :
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_sm_test, model.predict(x_sm_test), pos_label = 1)
  st.write('false_positive_rate :', false_positive_rate, 'true_positive_rate :', true_positive_rate)

  #le score AUC
  roc_auc_score_KNNsm = roc_auc_score(y_sm_test, model.predict(x_sm_test))
  st.write('score AUC sur données aprèes smote', roc_auc_score_KNNsm);

  #la courbe ROC
  fig = plt.figure();
  plt.plot(false_positive_rate, true_positive_rate);
  plt.plot([0, 1], ls="--");
  plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7");
  plt.ylabel('True Positive Rate');
  plt.xlabel('False Positive Rate');
  st.pyplot(fig);
  
  #Résultat :
  st.markdown("confirmation de la matrice de confusion")
  st.markdown("le classement des vrais positifs est moins bon que le classement des vrais négatifs")

  # MAE :
  MAE_KNNsm = mae(y_sm_test, y_pred_test_KNNsm)
  st.write("Mean Absolute Error' ou 'MAE' : " + str(MAE_KNNsm))
  st.markdown("la MAE est faible, ce qui corrobore avec la bonne performance du modèle")
