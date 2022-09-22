#imports
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
from imblearn import *
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import metrics
from PIL import Image
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.decomposition import PCA
import pickle
import joblib
from joblib import dump, load
import os


#chargements préliminaires nécessaires :
#création du jeu de données :
df_full = pd.read_csv("weatherAUS.csv")

#Création d'un jeu sans manquantes ni doublons : 
df = df_full.dropna()
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

#on élimine les variables enlevées dans le pipeline optimal : date plus les variables non explicatives non corrélées : :  .
df = df.drop('Date', axis = 1)
df = df.drop(['WindDir3pm','Temp9am','WindDir9am'], axis = 1)
df = df.drop(['day','month','Location', 'year'], axis = 1)

##-Renommer pour lisibilité les booléénnes : 
df['RainToday_encode'] = df['RainToday']
df['RainTomorrow_encode'] = df['RainTomorrow']
df = df.drop(labels = ['RainTomorrow', 'RainToday'], axis = 1)

#-encodage des restantes sur df:     
le = preprocessing.LabelEncoder()
df_nonencode = df#sauvegarde du df pour l'affichage ultérieur pour montrer affichage 
for var in df.select_dtypes(include='object').columns:#encodage
  df[var] = le.fit_transform(df[var])
df_encode = df#stockage pour une figure ultérieure

#création du jeu des explicatives x_sm et de cible y_sm : issus du pipeline optimal (dropna, sélection pearson, encodage et SMOTE):
x = df.drop('RainTomorrow_encode', axis = 1)
y = df['RainTomorrow_encode']
y = np.array(y)#reshape de y (explo des données et options de pipeline)
y.reshape(-1, 1)
y = y.astype(float)
smo = SMOTE()
x_sm, y_sm = smo.fit_resample(x, y)

df_sm = x_sm#pour figure
df_sm = df_sm.assign(RainTomorrow_encode = y_sm)#pour figure

#creation des jeux d'entrainement x_train et test x_test issus du pipeline optimal
x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.20, random_state=42)

#undersampling random (explo des données et options de pipeline)
rUs = RandomUnderSampler()
x_ru, y_ru = rUs.fit_resample(x, y)
df_ru = x_ru 
df_ru = df_ru.assign(RainTomorrow_encode = y_ru)

#normalisation (pour graphe de explo des données et options de pipeline)
x_norm = x#initialiser x_norm
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
name_columns_numerics = x_norm.select_dtypes(include=numerics).columns  
#créer, Entrainer et transformer directement les colonnes numériques de x_norm
scaler =  StandardScaler()
x_norm[name_columns_numerics] = scaler.fit_transform(x_norm[name_columns_numerics])


#Affichages des points clés de compréhension pour chaque étape du projet :

#Création du menu de choix à gauche et le choix est stocké sous la variable "rad": 
rad = st.sidebar.radio("Menu",["Introduction : Le projet et ses créateurs", 
                               "Exploration des données brutes", 
                               "Pipeline de préparation des données", 
                               "Evaluation de la performance des modèles pré-sélectionnés",
                               "Conclusion et perspectives"])
nuages_sidebar = Image.open('nuages_sidebar.jpg')
st.sidebar.image(nuages_sidebar)

#Si choix 1 :
if rad == "Introduction : Le projet et ses créateurs":
  st.title("Introduction : Le projet et ses créateurs")
  st.header("Le projet 'PyDiction'")
  if st.button('Cliquez ici pour afficher la description du projet et de cette application'):
    st.subheader("Titre du projet ")
    st.markdown("Le titre du projet 'PyDiction' est une synthèse des mots prédiction et python car l'on cherche ici à prédire la pluie et python a été employé pour cela ainsi que le Machine Learning. ")
    st.subheader("Définitions et objectifs du projet")
    st.markdown("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
    st.markdown("C'est un travail autour de la météorologie et du Machine Learning (ML). ")
    st.markdown("Les données sont issues de https://www.kaggle.com/jsphyg/weather-dataset-rattle-package et sont des données météorologiques de pluie, ensoleillement, température, pression, vent, humidité, pour plusieurs années et réparties sur 49 stations australiennes.")
    st.markdown("Le but a été de contruire un modèle de prédiction de la variable nommée 'RainTomorrow'. ")
    st.markdown("'RainTomorrow' représente la présence de pluie au lendemain d'un jour J (J + 1), sur un endroit quelconque en Australie, elle vaut tout simplement 1 si la pluie est > 1mm, 0 sinon.")
    st.subheader("Objectifs de cette application web ")
    st.markdown("Fournir de manière intéractive un résumé des étapes de construction du pipeline optimal de préparation des données et d'un modèle satisfisant pour prédire la pluie au lendemain sur un point du territoire australien." )
    st.subheader("Etapes du projet")
    diapo = Image.open('diapo.jpg')
    st.image(diapo, caption= "Etapes depuis les données météo jusqu'à la prédiction ")
    st.markdown("Vous pouvez découvrir les éléments clés du travail dans le cadre de ce projet en cliquant à gauche dans le menu")
    
  #Présentation des créateurs :
  st.header("Les créateurs") 
  if st.button('Cliquez ici pour découvrir les créateurs du projet'):
    Richard = Image.open('richard-modified.png')
    st.image(Richard, caption= "Richard, anciennement assistant de recherche en spectrométrie infrarouge, en reconversion dans la data science", width = 200)
    Nahim = Image.open('Nahim.png')
    st.image(Nahim, width = 200, caption="Nahim, anciennement ingénieur environnement et formateur en sciences, en reconversion dans l'informatique (data science, web) et les maths appliquées ")
    Adama = Image.open('Cthulhu.jpg')
    st.image(Adama, width = 200, caption="Mamadou, étudiant en physique, a rencontré un empechement. Depuis, il est perdu dans le temps et l'espace.")

    
#Si choix 2 :
elif rad == "Exploration des données brutes":
        
  #Exploration des données brutes :
  st.header("Exploration des données brutes : préparer la suite...")
  st.markdown("Avant la modélisation, une présélection de modèles à tester est classiquement faite en fonction de critères sur le jeu de données exploré, ainsi que des sources bibliographiques. Le traitement des données avant la modélisation peut se faire de différentes manières, soit obligatoirement : élimination ou remplacement des données manquantes et des doublons, encodage des catégorielles, éventuellement: normalisation, rééchantilonnage, réduction du nombre de variables. Afin de déterminer la méthode amenant à un jeu de qualité optimale et donc des performances optimales, une exploration thématique des données brutes est nécessaire.")
  
  #Nombre de données, et source :
  st.subheader("Source des données et nombre pour aider la préselction des modèles de ML :")   
  st.markdown("Le nombre de données est un critère pour la préselection de modèles et le choix du remplacement des manquantes. Les données sont présentes sur 49 stations australiennes, sur plusieurs années, et comprennent les informations journalières de : ensoleillement, humidité, vitesse et sens du vent, quantité de nuages, températures minimales et maximales etc.")
  st.write('Le nombre de lignes du jeu de données est :', 
           len(df_full), 
           'donc selon les critères usuels, le nombre de données est assez conséquent pour entrainer un modèle de prédiction et le rendre performant.')
  
  #Les variables explicatives, définitions et types  :
  st.subheader("Types des variables explicatives pour évaluer la nécessité de l'encodage :")    
  st.write("Affichons le contenu des données brutes pour repérer le nom des variables explicatives et leur type : ")
  st.dataframe(data=df_full)
  st.markdown("Les variables sont numériques ou catégorielles, il faudra donc encoder les catégorielles par la suite.")

  #Repérage des doublons et des manquantes : 
  st.subheader("Repérage des doublons et des valeurs manquantes éventuelles pour évaluer la nécessité de ces suppressions")    
  st.markdown("Ici il n'y a pas de doublons, mais des manquantes (voir ci-dessous) ")
  if st.checkbox("Cocher pour afficher les positions des données manquantes"):
    fig = plt.figure()  
    sns.heatmap(df_full.isnull(), cbar=False)
    st.pyplot(fig)  
  if st.checkbox("Cocher pour afficher le pourcentage de valeurs manquantes pour chacune des variables"):
    percent_missing_df_full = df_full.isnull().sum() * 100 / len(df_full)
    st.write(percent_missing_df_full)    

  #Etude de la distribution des variables pour étudier l'intérêt de la normalisation :
  st.subheader("Etude de la distribution des variables pour étudier l'intérêt de la normalisation")    
  st.markdown("Les modèles de ML demandent en entrée des données aux distributions normales, vérifions cette condition :")
  
  #Distribution des numériques :
  st.markdown("Etude de la distribution des variables numériques - boxsplot :")

  #création de sous dataframes
  df_minmaxtemp = df_full.loc[:, ['MinTemp', 'MaxTemp']] 
  df_wind = df_full.loc[:, ['WindSpeed9am', 'WindSpeed3pm']]
  df_humidity = df_full.loc[:, ['Humidity9am', 'Humidity3pm']]
  df_pressure = df_full.loc[:, ['Pressure9am', 'Pressure3pm']]
  df_cloud = df_full.loc[:, ['Cloud9am', 'Cloud3pm']]
  df_temp = df_full.loc[:, ['Temp9am', 'Temp3pm']]
  df_rainfall_evaporation = df.loc[:, ['Rainfall', 'Evaporation']]
  df_evaporation_sunshine = df_full.loc[:, ['Evaporation', 'Sunshine']]
  choice = st.selectbox('Sélectionnez les catégorielles à étudier :', 
                        ('températures min et max et vitesse du vent', 
                       'couverture nuageuse (matin et après midi) et températures (matin et après midi)',
                        'humidité et pressions (matin et après-midi)', 
                        'pluie-évaporation, et évaporation-ensoleillement'
                        ))
  if choice == 'températures min et max et vitesse du vent':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    sns.boxplot(data=df_minmaxtemp, color="red", ax=ax1)
    sns.boxplot(data=df_wind, color="green", ax=ax2 )
    ax1.set_title("températures min et max")
    ax2.set_title("vitesse du vent (9 pm et 3 am)")
    fig.set_tight_layout(True)
    st.pyplot(fig)  
  if choice == 'couverture nuageuse (matin et après midi) et températures (matin et après midi)':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    sns.boxplot(data=df_cloud, color="red", ax=ax1)
    sns.boxplot(data=df_temp, color="green", ax=ax2 )
    ax1.set_title("couverture nuageuse (9am, 3pm)")
    ax2.set_title("températures (9am, 3pm)")
    fig.set_tight_layout(True)
    st.pyplot(fig)
  if choice == 'humidité et pressions (matin et après-midi)':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    sns.boxplot(data=df_humidity, color="red", ax=ax1)
    sns.boxplot(data=df_pressure, color="green", ax=ax2 )
    ax1.set_title("humidité")
    ax2.set_title("pressions (9am, 3pm)")
    fig.set_tight_layout(True)
    st.pyplot(fig)
  if choice == 'pluie-évaporation, et évaporation-ensoleillement':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    sns.boxplot(data=df_rainfall_evaporation, color="red", ax=ax1)
    sns.boxplot(data=df_evaporation_sunshine, color="green", ax=ax2 )
    ax1.set_title("pluie - évaporation")
    ax2.set_title("évaporation - ensoleillement")
    fig.set_tight_layout(True)
    st.pyplot(fig)  
  st.markdown("Nous voyons que les distributions des variables numériques sont globalement gaussiennes. Nous avons posé l'hypothèse que les quelques outliers ne perturberont pas les entrainements des modèles vu leur nombre, et ils sont conservés pour permettre au modèle de s'adapter à de nouvelles données parfois extrêmes (changement climatique voir GIEC).")
  st.markdown("Etudions à présent la distribution de WindGustDir (direction des vents de la journée), en piechart pour voir les modalités et l'importance relative de chacune d'entre elles en nombre de valeurs correspondantes : ")
  
  #distribution de WindGustDir :
  fig_vents = Image.open('fig vents.png')
  st.image(fig_vents, width = 300, caption= "répartition de chaque valeur dans les catégories de WindGustDir, en piechart pour voir facilement les noms des directions simultanément")
    
  #Distribution de la variable cible pour évaluer l'intérêt des méthodes de rééchantillonnage: 
  st.subheader("Distribution de la variable cible pour évaluer l'intérêt des méthodes de rééchantillonnage :")
  if st.checkbox("Cocher pour afficher la distribution de RainTomorrow :"):
    fig = plt.figure(figsize=(3,3))
    sns.countplot(data = df_full, x = 'RainTomorrow')
    st.pyplot(fig)
  st.markdown("Les données sont déséquilibrées ce qui est classique en météorologie. Nous avons posé l'hypothèse que le rééquilibrage des données par rééchantillonnage sera utile sur les performances globales des modèles, la vérification de cette hypothèse est présentée par la suite.")
  
  #Préselection des modèles :
  st.subheader("Présélection de modèles suite à l'exploration des données" )
  st.markdown("Les modèles potentiellement adaptés selon la méthode de Scikit Learn et les études de ce type sont : KNN, régression logistique, arbre de décision et Random Forest  ")

#Si choix 3:
elif rad == "Pipeline de préparation des données": 
  st.header("Pipeline de préparation des données")
  
  #introduction de section : objectif
  st.subheader("L'objectif de la section, créer un pipeline optimal :")
  st.markdown("Afin d'obtenir un modèle aux performances optimales, il est généralement conseillé de préparer les données sur ces critères suivants. Voici donc les étapes généralement effectuées pour préparer au mieux les données, le pipeline optimal retenu et sa construction.") 

  #Traitement des données manquantes et des doublons:
  st.subheader("Traitement des manquantes et des doublons :")
  st.markdown("Les données manquantes doivent être enlevées car elles empêchent le bon fonctionnement des algorithmes. La meilleure option a été dans notre cas de choisir d'enlever toutes les données manquantes en une fois puisque l'imputation statistique n'a pas amené de meilleures performances des modèles et il faut par principe conserver le jeu de données le plus léger.")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  if st.checkbox("Cocher pour afficher le pourcentage de valeurs manquantes par colonnes :"):
    st.write(percent_missing_df)
  st.write("Affichons de nouveau le nombre de lignes", len(df), " ce nombre est réduit par rapport au départ : plus de 50% de suppression. Le score de prédiction étant le même avec les données manquantes enlevées ou traitées par imputation statistique, nous avons choisi de conserver le jeu de données réduit.")
  
  #Affichage de l'encodage :
  st.subheader("Encodage des catégorielles :")
  st.markdown("Une fois les données manquantes traitées, les variables catégorielles doivent être encodées pour réaliser la sélection des variables.")
  st.markdown("Les données ont été encodées par Label Encoder. L'encodage sous forme de données numériques doit etre vérifié.")
  if st.checkbox("Cocher pour afficher le tableau des données encodées :"):
    st.write(df_encode)
  
  #Sélection des variables par le test de Pearson :
  st.subheader("Sélection des variables, test de Pearson :")
  st.markdown("Il faut sélectionner les variables explicatives pour la modélisation avec un nombre final minimal sans perdre d'information pour éviter l'overfitting.")
  st.markdown("Pour cela, nous allons présenter la méthode de réduction de dimensions qui a donné les meilleures améliorations parmi les classiques testées (ACP, select kbest, pearson) :  afficher la matrice des corrélations et filtrer les variables non corréllées à la RainTomorrow afin de ne garder que des variables informatives :")
  heatmap, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  if st.checkbox("Cocher pour afficher la heatmap"):
    st.write(heatmap)
  st.markdown("Suppression des variables explicatives corréllées à moins de 5% à la cible selon le test de Pearson qui sont 'WindDir3pm','Temp9am','WindDir9am'") 
 
  st.subheader("Méthodes de normalisation, de réduction de dimensions et de rééchantillonnage")
  st.markdown("Une fois les manquantes et les doublons enlevés et les données encodées, les méthodes de préparations classiques des données sont mises en oeuvre afin d'assurer de bons résultats en machine learning sur un jeu de données déséquilibré au niveau de la variable cible : méthodes de normalisation (pour éviter les problèmes liés aux échelles trop différentes), réduction de dimension (pour limiter le surapprentissage sur des données inutiles) et de rééchantillonnage (pour compenser le déséquilibre de répartition évoqué précédemment) ")

  #Sélection de la méthode de rééchantillonnage et impact :
  choice = st.selectbox("Voici un affichage de l'effet des rééchantillonnages sur la distribution des valeurs dans la cible, optionnel en pratique mais utile ici pour comprendre sélectionnez la méthode de rééchantillonnage que vous voulez appliquer aux données :", ('Aucun rééchantillonnage', 'Undersampling', 'OverSampling SMOTE'))
  
  #afficher le choix sélectionné :
  if choice == 'Aucun rééchantillonnage':
    #nothing
    x_def = x
    y_def = y
    
    st.write("le nombre de lignes reste inchangé :", len(df))
    
    if st.checkbox("Cocher pour afficher la distribution de RainTomorrow :"):
      fig = plt.figure(figsize=(3,3))
      sns.countplot(data = df, x = 'RainTomorrow_encode')
      st.pyplot(fig)
        
  elif choice == 'Undersampling':
    #affectation de x et y
    x_def = x_ru
    y_def = y_ru
    
    st.write("le nombre de lignes diminue :", len(x_ru))
    
    if st.checkbox("Cocher pour afficher la distribution de RainTomorrow :"):
      fig = plt.figure(figsize=(3,3))
      sns.countplot(data = df_ru, x = 'RainTomorrow_encode')
      st.pyplot(fig)
        
  elif choice == 'OverSampling SMOTE':
    #affectation de x et y
    x_def = x_sm
    y_def = y_sm   
    
    st.write("le nombre de lignes augmente :", len(x_sm))
    
    if st.checkbox("Cocher pour afficher la distribution de RainTomorrow :"):
      fig = plt.figure(figsize=(3,3))
      sns.countplot(data = df_sm, x = 'RainTomorrow_encode')
      st.pyplot(fig)
      
  
  if st.checkbox("Cocher pour afficher notre conclusion quant au resampling :"):
    st.markdown("Un rééchantillonnage SMOTE a été retenu pour la préparation optimale de ce jeu puisque nous avons de meilleures performances avec. ")
  
  st.markdown("La question de la normalisation s'est aussi posée")
  choice2 = st.selectbox("Voici une visualisation montrant que la normalisation a bien un effet ici", ('Aucune normalisation','StandardScaler'))
  #displaying the selected option
  st.write('Vous avez sélectionné :', choice2)
  
  if choice2 == 'Aucune normalisation':
    #affectation de x et y
    
    df_minmaxtemp = df.iloc[:, 1:3]    
    fig = plt.figure(figsize=(3,3))
    sns.boxplot(data=df_minmaxtemp, color="red")
    #ax1.set_title("températures min et max")
    st.pyplot(fig)  
  
  elif choice2 == 'StandardScaler':
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    name_columns_numerics = x.select_dtypes(include=numerics).columns
    
    x_norm_minmaxtemp = x_norm.iloc[:, 1:3]
    
    fig = plt.figure(figsize=(3,3))
    sns.boxplot(data=x_norm_minmaxtemp, color="red")
    #ax1.set_title("températures min et max")
    st.pyplot(fig)  
   
  if st.checkbox("Cocher pour afficher notre conclusion quant à la normalisation et aux méthodes de réduction de dimensions:"):
    st.markdown("Les méthodes de normalisation ou de réduction de dimensions n'ayant pas amené d'amélioration des résultats de performances des modèles, nous ne les avons donc pas conservées. ")
    
#Si choix 4 :
if rad == "Evaluation de la performance des modèles pré-sélectionnés":
  st.header("Evaluation de la performance globale des modèles préselectionnés")
  st.markdown("Comme vu précédemment, le pipeline optimal est : conserver les variables corréllées à RainTomorrow puis méthode de rééchantillonnage SMOTE.")
  st.markdown("Puis le jeu de données est découpé en jeu de test et d'entrainement à hauteur de 20% et 80% respectivement afin de pouvoir évaluer les modèles sur le jeu test.")     
  st.markdown("Pour chacun des 4 modèles à tester selon nos recherches et la méthode de Scikit Learn, le modèle est optimisé par gridsearch puis entrainé sur le jeu traité par le pipeline optimal puis évalué")
  st.markdown("Nous commençons par KNN qui est le modèle sélectionné au final, les autres modèles sont évalués sur les pages suivantes (voir le menu)")
  
  model_choice = st.selectbox('choisir le modèle à charger', ('KNN optimisé', 'DTC optimisé', 'log reg optimisé', 'RFC optimisé'))
  st.subheader("vous avez choisi de charger")
  st.subheader(model_choice)

  #sauvegarde des noms des modèles pour rappel : 
  #knn = KNeighborsClassifier(metric='manhattan', n_neighbors=26, weights='distance') #mettre ici le meilleur nbr_voisins trouvé plus haut
  #dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7, min_samples_leaf = 40, random_state = 123)
  #lr = LogisticRegression(C=0.01, penalty= 'l2')
  #rfc = RandomForestClassifier(max_depth = 8, n_estimators = 200, criterion = 'gini', max_features = 'sqrt')
  
  model = model1#initialisation du modèle (il faut un premier choix initial, changeable ensuite)
  filename1 = "KNNbest_pipeline_opti.joblib"
  model1 = joblib.load(filename1)

  filename2 = "DTCbest_pipeline_opti.joblib"
  model2 = joblib.load(filename2)

  filename3 = "LogRegbest_pipeline_opti.joblib"
  model3 = joblib.load(filename3)

  filename4 = "RForest_pipeline_opti.joblib"
  model4 = joblib.load(filename4)
  
  if model_choice == 'KNN optimisé':
    model = model1
  
  elif model_choice == 'DTC optimisé':
    model = model2
 
  elif model_choice == 'log reg optimisé':
    model = model3
   
  elif model_choice == 'RFC optimisé':
    model = model4

  st.markdown("Maintenant que l'entrainement du modele est chargé, étudions les indicateurs de performance du modèle sélectionné :")
  
  ##Précision et f1-score : sur x_train (jeu entrainement issu de pipeline optimal) et x_test (jeu test issu du pipeline optimal)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test) 
  index_choice = st.selectbox('Choisissez une métrique ?',('accuracy','F1-score','AUC et ROC Curve'))

  if index_choice == 'accuracy':      
    acc_train  = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test, y_pred_test)
    st.write("accuracy sur jeu d'entrainement par le pipeline optimal : ", acc_train, "accuracy sur jeu test par le pipeline optimal : :", acc_test)

  elif index_choice == 'F1-score' :
    f1score_train = f1_score(y_train, y_pred_train, average='macro')
    f1score_test = f1_score(y_test, y_pred_test, average='macro')
    st.write("F1score_train : ", f1score_train, "F1score_test : ", f1score_test)

  elif index_choice == 'AUC et ROC Curve' :
    st.markdown('Imprimons à présent la courbe ROC de ce modèle : ')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict(x_test), pos_label = 1)
    roc_auc_score = roc_auc_score(y_test, model.predict(x_test))

    #la courbe ROC
    fig = plt.figure();
    plt.plot(false_positive_rate, true_positive_rate);
    plt.plot([0, 1], ls="--");
    plt.plot([0, 0], [1, 0] , c=".7"), 
    plt.plot([1, 1] , c=".7");
    plt.ylabel('True Positive Rate');
    plt.xlabel('False Positive Rate');
    st.pyplot(fig);
    st.write('Le score AUC est de', roc_auc_score, 'pour intepréter : plus il est proche de 1 plus le modèle est précis, plus il est proche 0.5 moins le modèle est précis.');
    st.markdown('Le score AUC ici est donc acceptable. ')
    st.markdown("Le classement des vrais positifs est cependant moins bon que le classement des vrais négatifs")
    st.markdown('Les scores d accuracy (précision globale) et de f1-score (sensible à la précision de prédiction de chaque classe) sur les jeux d entrainement et de test sont :  ')

  #elif index_choice == 'MAE' : La MAE a été désactivée, car elle ne sert à rien dans le cadre de problèmes de classifications. On la laisse donc en tant que trace. La MAE n'a pas rentré en compte dans le choix du meilleur modèle, elle a été rajouté par erreur en deuxième temps (coquille).
  #  MAE = mae(y_test, y_pred_test)
  #  st.write("La 'Mean Absolute Error' ou 'MAE' est de : " + str(MAE), ', plus elle est basse plus le modèle est précis. Notre modèle a donc ici une précision correcte, ce paramètre d erreur est cohérent et confirme le score de précision. ')

  #résultats :
  st.markdown("Les prédictions sont plutôt bonnes !")
  
if rad == "Conclusion et perspectives":
  
  st.subheader("Conclusion")
  
  st.markdown("Nous avons pu sélectionner les variables les plus pertinentes grâce aux tests statistiques. Des modèles de classification simples offrent des performances similaires à celles offertes par des modèles ensemblistes.")
  st.markdown("Au vu de la répartition de la population cible, un resampling par oversampling SMOTE est nécessaire et son efficacité a été montrée. Ainsi, nous confirmons notre capacité à prédire Rain-Tomorrow avec une marge d'erreur acceptable.")
  st.markdown("Au final, deux algorithmes offrent des performances satisfaisantes à la fois en terme de métriques et de temps d/'execution sont: KNN et Régression Logistique.") 
  st.markdown("Finalement, nous avons pu recourir au stockage des entrainements des différents via joblib, appelés gràce à ce script exécuté localement, au prix de temps d'execution allongés, de problèmes de connexion, et d'instabilité du streamlit")
  st.markdown("A titre d'indication, nous listons ci dessous les métriques du KNN.")
    
  st.markdown("acc_train :  1.0, acc_test : 0.86.")
  st.markdown("F1score_train :  1.0 F1score_test : 0.86.")
  st.markdown("Mean Absolute Error' ou 'MAE' : 0.13.")
  st.markdown("l'AUC est de : 0.87.")

  st.markdown("Les performances sont tout à fait acceptables, et nous permettent de répondre à la problématique")
  
  st.subheader("Limites")

  st.markdown("Dans l'optique d'enrichir les capacités de notre modèle, nous avons tenté de créer la variable RainIn3Days, en nous servant de Rain_Today.")
  st.markdown("Pour cela, nous avons tenté de processer la variable de RainIn3Days à partir de RainToday, au moyen d’une boucle prenant en compte la date et la localisation (en effet, il y a plusieurs stations météos, et seulement 256 dates qui ne sont pas en doublons : cela signifie qu’il y a plusieurs bulletins météos émis le même jour par ces stations).")
  st.markdown("Malheureusement, nous n'avons pas pu mener cette étude à son terme par manque de temps notamment suite au départ d’une personne de l’équipe.")

  st.markdown("Une autre question posée était la question des outliers. Nous n’avons pas vérifié l’hypothèse de l’élimination des outliers en entier")
  st.markdown("Nous aurions pu aussi tenter de filter le jeu de données selon ses quartiles. Malheureusement, nous n'avons pas réussi à résoudre nos problèmes de code à temps.")
  st.markdown("Même si nous sommes convaincus que les résultats ne seraient pas améliorés par la suppression de ces extrêmes, la rigueur indique tout de même de vérifier cette hypothèse dans une étude ultérieure.")

  st.markdown("Nous aurions pu utiliser d’autres modèles tels que les réseaux de neurones, et les méthodes de séries temporelles. A deux personnes au lieu de trois et au vu des alternatives et du nombre de modèles testés, nous sommes satisfaits de la quantité de résultats.")

  
  if st.checkbox("Cocher pour afficher la surprise !:"):
    st.markdown("Un rééchantillonnage SMOTE a été retenu pour la préparation optimale de ce jeu puisque nous avons de meilleures performances avec... sérieusement, vous vous attendiez à quoi ? ")



