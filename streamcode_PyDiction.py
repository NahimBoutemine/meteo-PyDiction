#Première partie : imports et création des variables (jeux de données, indicateurs...) à afficher :

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
from PIL import Image
from sklearn.preprocessing import StandardScaler

#création du jeu de données :

df = pd.read_csv("weatherAUS.csv")
df = df.dropna()
df = df.drop_duplicates()

#Encodage des données et stockage des variables à afficher dans l'application :

#-Traitement de la variable 'date' :
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
##-Renommer pour lisibilité les booléénnes : 
df['RainToday_encode'] = df['RainToday']
df['RainTomorrow_encode'] = df['RainTomorrow']
df = df.drop(labels = ['RainTomorrow', 'RainToday'], axis = 1)
#encodage des restantes:     
le = preprocessing.LabelEncoder()
df_nonencode = df#sauvegarde du df pour l'affichage ultérieur
for var in df.select_dtypes(include='object').columns:
  df[var] = le.fit_transform(df[var])
df_encode = df#stocjage à ce stade du df aux variables encodées

df.drop(['WindDir3pm','Temp9am','WindDir9am'], axis = 1) 

#Affichages des étapes du projet et des  points clés à partir des variables stockées plus haut :

#Création du menu de choix à gauche et le choix est stocké sous la variable "rad": 
rad = st.sidebar.radio("Menu",["Introduction : Le projet et ses créateurs", "Préparation des données - partie 1 : élimination des manquantes et encodage des données", "Préparation des données - partie 2 : Méthodes de normalisation, de réduction de dimensions et de rééchantillonnage", "Machine Learning", "Conclusion et perspectives"])

#Si choix 1 :
if rad == "Introduction : Le projet et ses créateurs":
  st.title("Introduction : Le projet et ses créateurs")
  st.header("Le projet 'PyDiction'")
  if st.button('Cliquer ici pour afficher la description du projet et de cette application'):
    kangourou = Image.open('kangoufun.jpg')
    st.image(kangourou, caption=' ')
    st.markdown("Le titre du projet 'PyDiction' est une synthèse des mots prédiction et python car l'on cherche ici à prédire la pluie et python a été employé pour cela ainsi que le Machine Learning. ")
    st.markdown("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
    st.markdown("C'est un travail autour de la météorologie et du Machine Learning (ML). ")
    diapo = Image.open('diapo.jpg')
    st.image(diapo, caption="Etapes depuis les données météo jusqu'à la prédiction ")
    st.markdown("Les données sont issues de https://www.kaggle.com/jsphyg/weather-dataset-rattle-package et sont des données météorologiques de pluie, ensoleillement, température, pression, vent, humidité, pour plusieurs années et réparties sur 49 stations australiennes.")
    st.markdown("Le but a été de contruire un modèle de prédiction de la variable nommée 'RainTomorrow'. ")
    st.markdown("'RainTomorrow' représente la présence de pluie au lendemain d'un jour J (J + 1) sur un point du territoire australien, elle vaut tout simplement 1 si la pluie est > 1mm, 0 sinon.")
    st.markdown("Cette application streamlit vise à montrer les étapes ayant permis de conclure sur une méthode optimale de préparation des données et sur un modèle au taux de prédiction satisfisant pour ce type de données." )

  #Présentation des créateurs :
  st.header("Les créateurs") 
  if st.button('Cliquer ici pour découvrir les créateurs du projet'):
    Richard = Image.open('richard.jpg')
    st.image(Richard, caption="Richard, anciennement assistant de recherche en spectrométrie infrarouge, en reconversion dans la data science", width = 200)
    Nahim = Image.open('Nahim.png')
    st.image(Nahim, width = 200, caption="Nahim, anciennement ingénieur environnement et formateur en sciences, en reconversion dans l'informatique (data science, web) et les maths appliquées ")

elif rad == "Description du jeu de données":
  
  #Présentation du jeu de données
  st.header("Présentation et exploration des données")
  st.markdown("Les données sont présentes sur 49 stations australiennes, sur plusieurs années, et comprennent les informations de : ensoleillement, humidité, vitesse et sens du vent, quantité de nuages, températures minimales et maximales etc.")
  st.markdown("Elles ne comprennent pas de doublons mais contiennent des données manquantes. ")
  st.markdown("La pluie est considérée comme présente au jour J si elle est strictement supérieure à 1mm. ")
  
  if st.button("Cliquer ici pour découvrir la suite de l'exploration des données brutes"):
    st.markdown("Voici le contenu des données, vous pouvez y voir déjà les noms des variables ainsi que la cible, RainTomorrow :")
    st.write(df)
    st.markdown("Le nombre de lignes est:")
    st.write(len(df))
    st.markdown("S'il existe des valeurs manquantes, elles sont à enlever pour le bon déroulement de la modélisation, de même que les doublons. ")
    st.markdown("Affichons le pourcentage de ces fameuses valeurs manquantes, et ce pour chacune des variables :")
    percent_missing_df = df.isnull().sum() * 100 / len(df)
    st.write(percent_missing_df)
    st.markdown("Ainsi, les valeurs manquantes sont enlevées car, même si en général elles sont remplacées par une valeur (imputation statistique), selon notre expérience dans ce cas de prédiction cela ne fait que rajouter du temps de calcul")
    st.markdown("A présent le nombre de données manquantes : on peut voir à présent qu'il n'y en a plus!")
    percent_missing_df = df.isnull().sum() * 100 / len(df)
    st.write(percent_missing_df)
    st.markdown("Affichons le pourcentage de valeurs non en doublon pour vérifier qu'elles ont été supprimées:")
    percentage_dupli = df.duplicated(keep=False).value_counts(normalize=True) * 100
    st.write(percentage_dupli)
    st.markdown("Affichons de nouveau le nombre de lignes, ce nombre est réduit par rapport au départ : plus de 50% de suppression. Le score de prédiction étant le même avec les données manquantes enlevées ou traitées par imputation statistique, nous avons choisi de conserver le jeu de données réduit.")
    st.write(len(df))
    
    #afficher la répartition des valeurs dans la cible:
    st.markdown("Affichons la répartition des valeurs dans les catégories de la variable cible:")
    fig = plt.figure()
    sns.countplot(data = df, x = 'RainTomorrow')
    st.pyplot(fig)
    st.markdown("Les données sont déséquilibrées ce qui est classique en météorologie. Nous avons posé l'hypothèse que le rééquilibrage des données par rééchantillonnage sera utile sur les performances globales des modèles, les effets rééls de ce rééchantillonnage sont présntés ensuite et en conclusion.")

elif rad == "Préparation des données - partie 1 : élimination des manquantes et encodage des données":
  st.markdown("Les données sont encodées. Les données avant encodage sont de ce type : ")
  st.write(df_nonencode)
  st.markdown("Les données encodées par Label Encoder sont de cette forme : ")
  st.write(df_encode)
  
  #heatmap
  st.markdown("A présent, il faut sélectionner les variables explicatives pour la modélisation.")
  st.markdown("Pour cela, nous allons afficher la matrice des corrélations")
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)
  st.markdown("Suppression des variables explicatives corréllées à moins de 5% à la cible selon le test de Pearson qui sont 'WindDir3pm','Temp9am','WindDir9am'") 


  #(méthode courante d'évaluation):
  y = df['RainTomorrow_encode']
  x = df.drop('RainTomorrow_encode', axis = 1)
  

elif rad == "Préparation des données - partie 2 : Méthodes de normalisation, de réduction de dimensions et de rééchantillonnage":
  
  st.markdown("Le jeu de données est ensuite découpé en jeu de test et d'entrainement à hauteur de 20% et 80% respectivement afin de pouvoir évaluer les modèles sur le jeu test.") 
  st.markdown(" Une fois les manquantes et les doublons enlevés et les données encodées, les méthodes de préparations classiques des données afin d'assurer de bons résultats en machine learning sur un jeu de données déséquilibré au niveau de la variable cible : méthodes de normalisation (pour éviter les problèmes liés aux échelles trop différentes), réduction de dimension (pour limiter le surapprenisage sur des données inutiles) et de rééchantillonnage (pour compenser le déséquilibre de répartition évoqué précédemment) ")
  st.markdown("Un rééchantillonnage SMOTE a été retenu pour la préparation optimale de ce jeu puisque nous avons de meilleures performances avec. ")
  st.markdown("Les méthodes de normalisation ou de réduction de dimensions n'ayant pas amené d'améloration des résultats de performances des modèles, nous ne les avons donc pas conservées. ")
 
  #selection de la méthode de rééchantillonage :
  choice = st.selectbox('Sélectionnez la méthode de rééchantillonnage que vous voulez appliquer aux données :', ('Aucune', 'Undersampling', 'OverSampling SMOTE'))
  #afficher le choix sélectionné :
  st.write('Vous avez sélectionné :', choice)
  choice = str(choice)
  



elif rad == "Conclusion et perspectives":
  st.header("Conclusion et perspectives")
  st.markdown("Nous avons pu sélectionner les variables les plus pertinentes grâce aux tests statistiques. Des modèles de classification simples offrent des performances similaires à celles offertes par des modèles ensemblistes. Au vu de la répartition de la population cible, un resampling par oversampling SMOTE est nécessaire et son efficacité a été montrée. Ainsi, nous confirmons notre capacité à prédire Rain-Tomorrow avec une marge d'erreur acceptable.")
  st.markdown("acc_train :  1.0, acc_test : 0.87.")
  st.markdown("F1score_train :  1.0 F1score_test : 0.87.")
  st.markdown("Mean Absolute Error' ou 'MAE' : 0.13.")
  st.markdown("l'AUC est de : 0.87.")


  st.text("Limites")

  st.markdown("Tentative de création et prédiction de RainIn3Days")

  st.markdown("Dans l'optique d'enrichir les capacités de notre modèle, nous avons tenté de créer la variable RainIn3Days, en nous servant de Rain_Today.")
  st.markdown("Pour cela, nous avons tenté de processer la variable de RainIn3Days à partir de RainToday, au moyen d’une boucle prenant en compte la date et la localisation (en effet, il y a plusieurs stations météos, et seulement 256 dates qui ne sont pas en doublons : cela signifie qu’il y a plusieurs bulletins météos émis le même jour par ces stations).")
  st.markdown("Malheureusement, nous n'avons pas pu mener cette étude à son terme par manque de temps notamment suite au départ d’une personne de l’équipe.")

  st.markdown("Traitement des outliers : nous n’avons pas vérifié l’hypothèse de l’élimination des outliers en entier, même si nous sommes convaincus que les résultats ne seraient pas améliorés par la suppression de ces extrêmes ainsi que par nos raisons de ce choix. La rigueur indique tout de même de vérifier cette hypothèse dans une étude ultérieure.")

  st.markdown("Autres modèles : Nous aurions pu utiliser d’autres modèles tels que les réseaux de neurones, les méthodes de séries temporelles. A deux personnes au lieu de trois et au vu des alternatives et du nombre de modèles testés, nous sommes satisfaits de la quantité de résultats.  De même pour la pluie dans trois jours.")










