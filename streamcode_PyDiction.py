#Première partie : imports et création des variables (jeux de données, indicateurs...) à afficher :

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

#chargements préliminaires nécessaires :
#création du jeu de données :
df = pd.read_csv("weatherAUS.csv")
df_full = df
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

#-encodage des restantes:     
le = preprocessing.LabelEncoder()
df_nonencode = df#sauvegarde du df pour l'affichage ultérieur
for var in df.select_dtypes(include='object').columns:
  df[var] = le.fit_transform(df[var])
df_encode = df#stocjage à ce stade du df aux variables encodées

#suppression des variables non explicatives (fonction du test de pearson, voir rapport et expliqué dans le streamlit également pour la présentation) :
df.drop(['WindDir3pm','Temp9am','WindDir9am'], axis = 1)

#Préparation du split :
#(méthode courante d'évaluation):
y = df['RainTomorrow_encode']
x = df.drop('RainTomorrow_encode', axis = 1)

#Affichages des points clés des étapes du projet :

#Création du menu de choix à gauche et le choix est stocké sous la variable "rad": 
rad = st.sidebar.radio("Menu",["Introduction : Le projet et ses créateurs", "Exploration des données brutes", "Préparation des données - partie 1 : élimination des manquantes, encodage et sélection des variables explicatives", "Préparation des données - partie 2 : Méthodes de normalisation, de réduction de dimensions et de rééchantillonnage", "Machine Learning", "Conclusion et perspectives"])

#Si choix 1 :
if rad == "Introduction : Le projet et ses créateurs":
  st.title("Introduction : Le projet et ses créateurs")
  st.header("Le projet 'PyDiction'")
  if st.button('Cliquez ici pour afficher la description du projet et de cette application'):
    kangourou = Image.open('kangoufun.jpg')
    st.image(kangourou, caption=' ')
    st.markdown("Le titre du projet 'PyDiction' est une synthèse des mots prédiction et python car l'on cherche ici à prédire la pluie et python a été employé pour cela ainsi que le Machine Learning. ")
    st.markdown("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
    st.markdown("C'est un travail autour de la météorologie et du Machine Learning (ML). ")
    diapo = Image.open('diapo.jpg')
    st.image(diapo, caption= "Etapes depuis les données météo jusqu'à la prédiction ")
    st.markdown("Les données sont issues de https://www.kaggle.com/jsphyg/weather-dataset-rattle-package et sont des données météorologiques de pluie, ensoleillement, température, pression, vent, humidité, pour plusieurs années et réparties sur 49 stations australiennes.")
    st.markdown("Le but a été de contruire un modèle de prédiction de la variable nommée 'RainTomorrow'. ")
    st.markdown("'RainTomorrow' représente la présence de pluie au lendemain d'un jour J (J + 1) sur un endroit quelconque en Australie, elle vaut tout simplement 1 si la pluie est > 1mm, 0 sinon.")
    st.markdown("Cette application streamlit vise à montrer de manière intéractive un résumé des étapes ayant permis de conclure sur un pipeline optimal de préparation des données et sur un modèle satisfisant pour prédire RainTomorrow sur un point du territoire australien." )

  #Présentation des créateurs :
  st.header("Les créateurs") 
  if st.button('Cliquez ici pour découvrir les créateurs du projet'):
    Richard = Image.open('richard-modified.png')
    st.image(Richard, caption= "Richard, anciennement assistant de recherche en spectrométrie infrarouge, en reconversion dans la data science", width = 200)
    Nahim = Image.open('Nahim.png')
    st.image(Nahim, width = 200, caption="Nahim, anciennement ingénieur environnement et formateur en sciences, en reconversion dans l'informatique (data science, web) et les maths appliquées ")

#Si choix 2 :
elif rad == "Exploration des données brutes":
  #Exploration des données brutes :
  st.header("Présentation et exploration des données")
  
  #Nombre de données,  définitions et types des variables :
  st.subheader("Source et nombre de données :")    
  st.markdown("Les données sont présentes sur 49 stations australiennes, sur plusieurs années, et comprennent les informations journalières de : ensoleillement, humidité, vitesse et sens du vent, quantité de nuages, températures minimales et maximales etc.")
  st.write('Le nombre de lignes du jeu de données est :', 
           len(df_full), 
           'donc selon les critères usuels, le nombre de données est assez conséquent pour entrainer un modèle de prédiction et le rendre perfo
  st.markdown("La pluie est considérée comme présente au jour J si elle est strictement supérieure à 1mm. ")
  st.subheader("variables : définitions et leurs types  :")    
  st.write('Affichons le contenu des données brutes pour repérer le nom des variables explicatives et leur type : ', df_full)
  rmant.')
  st.markdown("Les variables sont numériques ou catégorielles, il faudra donc encoder les catégorielles par la suite (condition nécessaire pour les algoritmes de ML).")

  #Repérage des doublons et des manquantes : 
  st.subheader("Repérage des doublons et des valeurs manquantes")    
  st.markdown("Par principe en ML, s'il existe des valeurs manquantes ou en doublons, elles sont à enlever pour le bon déroulement de la modélisation, il faut donc déjà vérifier leur existence: ")
  st.markdown("Ici il n'y a pas de doublons, mais des manquantes (voir ci-dessous) ")
  st.markdown("Pourcentage de  valeurs manquantes pour chacune des variables :")
  percent_missing_df_full = df_full.isnull().sum() * 100 / len(df)
  st.write(percent_missing_df_full)    

  #Distribution des variables pour vérifier la normalité :
  st.subheader("Etude de la distribution des variables pour vérifier la normalité")    
  
  #cible : 
  st.markdown("Variable cible, répartition des valeurs dans ses catégories :")
  fig = plt.figure(figsize=(3,3))
  sns.countplot(data = df_full, x = 'RainTomorrow')
  st.pyplot(fig)
  st.markdown("Les données sont déséquilibrées ce qui est classique en météorologie. Nous avons posé l'hypothèse que le rééquilibrage des données par rééchantillonnage sera utile sur les performances globales des modèles, les effets rééls de ce rééchantillonnage sont présntés ensuite et en conclusion.")

  #catégorielles :
  st.markdown("Variables catégorielles par boxsplot :")

  #création de sous dataframes
  df_minmaxtemp = df.iloc[:, 1:3]
  df_wind = df.iloc[:, 10:12]
  df_humidity = df.iloc[:, 12:14]
  df_pressure = df.iloc[:, 14:16]
  df_cloud = df.iloc[:, 16:18]
  df_temp = df.iloc[:, 18:20]
  df_rainfall_evaporation = df.iloc[:, 3:5]
  df_evaporation_sunshine = df.iloc[:, 4:6]
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
    ax2.set_title("vitesse du vent (9pm et 3 am)")
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
  st.markdown("Nous voyons que les distributions des variables sont globalement gaussiennes. Nous avons posé l'hypothèse que les quelques outliers ne perturberont pas les entrainements des modèles vu leur nombre, et sont conervés pour permettre au modèle de s'adapter à de nouvelles données parfois extrêmes (changement climatique voir GIEC).")
 
#Si choix 3:
elif rad == "Préparation des données - partie 1 : élimination des manquantes, encodage et sélection des variables explicatives":  

  st.header("Pipeline de préparation des données partie 1 : Traitement des manquantes, des doublons, encodage et sélection des variables explicatives :")

  #Traitement des manquantes :
  st.subheader("Traitement des manquantes :")
  st.markdown("Les données manquantes doivent être enlevées car elles empêchent le bon fonctionnement des algorithmes. La meilleure option a été dans notre cas de choisir d'enlever toutes les données manquantes en une fois puisque l'imputation statistique n'a pas amené de meilleures performances des modèles et il faut par principe conserver le jeu de données le plus léger.")
  st.markdown("Les valeurs manquantes sont ici enlevées car, même si en général elles sont remplacées par une autre valeur (imputation statistique), selon notre expérience dans ce cas de prédiction cela ne fait que rajouter du temps de calcul")
  st.markdown("A présent affichons ci-dessous le pourcentage de données manquantes par colonne : on peut voir qu'il n'y en a plus!")
  percent_missing_df = df.isnull().sum() * 100 / len(df)
  if st.checkbox("Cocher pour afficher le pourcentage de valeurs manquantes par colonnes :"):
    st.write(percent_missing_df)
  st.markdown("Affichons de nouveau le nombre de lignes", len(df)," ce nombre est réduit par rapport au départ : plus de 50% de suppression. Le score de prédiction étant le même avec les données manquantes enlevées ou traitées par imputation statistique, nous avons choisi de conserver le jeu de données réduit.")
  st.markdown("Une fois ces données manquantes enlevées, les données doivent être encodées pour réaliser le test de pearson et donc la sélection des variables. Avant encodage elles sont ainsi :")

  #Affichage de l'encodage :
  st.subheader("Encodage des catégorielles :")
  st.markdown("Les données encodées par Label Encoder sont de cette forme (ci-dessous),nous vérifions qu'elles sont bien toutes numériques: ")
  if st.checkbox("Cocher pour afficher le tableau des données encodées :"):
    st.write(df_encode)
  
  #Sélection des variables par le test de Pearson :
  st.subheader("Sélection des variables : par le test de Pearson :")
  st.markdown("Il faut sélectionner les variables explicatives pour la modélisation avec un nombre final minimal sans perdre d'information pour éviter l'overfitting.")
  st.markdown("Pour cela, nous allons afficher la matrice des corrélations et filtrer les variables non corréllées à la RainTomorrow afin de ne garder que des variables informatives :")
  heatmap, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  if st.checkbox("Cocher pour afficher la heatmap"):
    st.write(heatmap)
  st.markdown("Suppression des variables explicatives corréllées à moins de 5% à la cible selon le test de Pearson qui sont 'WindDir3pm','Temp9am','WindDir9am'") 
 
#Si choix 4 :
elif rad == "Préparation des données - partie 2 : Méthodes de normalisation, de réduction de dimensions et de rééchantillonnage":  
  st.markdown("Le jeu de données est ensuite découpé en jeu de test et d'entrainement à hauteur de 20% et 80% respectivement afin de pouvoir évaluer les modèles sur le jeu test.") 
  st.markdown(" Une fois les manquantes et les doublons enlevés et les données encodées, les méthodes de préparations classiques des données afin d'assurer de bons résultats en machine learning sur un jeu de données déséquilibré au niveau de la variable cible : méthodes de normalisation (pour éviter les problèmes liés aux échelles trop différentes), réduction de dimension (pour limiter le surapprenisage sur des données inutiles) et de rééchantillonnage (pour compenser le déséquilibre de répartition évoqué précédemment) ")
  st.markdown("Un rééchantillonnage SMOTE a été retenu pour la préparation optimale de ce jeu puisque nous avons de meilleures performances avec. ")
  st.markdown("Les méthodes de normalisation ou de réduction de dimensions n'ayant pas amené d'améloration des résultats de performances des modèles, nous ne les avons donc pas conservées. ")
 
  #selection de la méthode de rééchantillonage :
  choice = st.selectbox('Sélectionnez la méthode de rééchantillonnage que vous voulez appliquer aux données :', ('Aucun rééchantillonage', 'Undersampling', 'OverSampling SMOTE'))
  #afficher le choix sélectionné :
  st.write('Vous avez sélectionné :', choice)
  choice = str(choice)
  
  if choice == 'Aucun rééchantillonage':
    #tenter le curseur glissant pour le split ? ou on reste à 20%
    #selection de la normalisation
        
    #afficher la répartition des valeurs dans la cible:
    st.markdown("Affichons la répartition des valeurs dans les catégories de la variable cible:")
    fig = plt.figure()
    sns.countplot(data = df, x = 'RainTomorrow_encode')
    st.pyplot(fig)
    st.markdown("Les données sont déséquilibrées ce qui est classique en météorologie. Nous avons posé l'hypothèse que le rééquilibrage des données par rééchantillonnage sera utile sur les performances globales des modèles, les effets rééls de ce rééchantillonnage sont présntés ensuite et en conclusion.")

  elif choice == 'Undersampling':
    rUs = RandomUnderSampler()
    x_ru, y_ru = rUs.fit_resample(x, y)
    #affectation de x et y
    x = x_ru
    y = y_ru
    
    #afficher la répartition des valeurs dans la cible:
    st.markdown("Affichons la répartition des valeurs dans les catégories de la variable cible:")
    fig = plt.figure()
    sns.countplot(data = df, x = 'RainTomorrow_encode')
    st.pyplot(fig)
    st.markdown("Les données sont déséquilibrées ce qui est classique en météorologie. Nous avons posé l'hypothèse que le rééquilibrage des données par rééchantillonnage sera utile sur les performances globales des modèles, les effets rééls de ce rééchantillonnage sont présntés ensuite et en conclusion.")

    
  elif choice == 'OverSampling SMOTE':
    st.write('Vous avez sélectionné :', choice)
    smo = SMOTE()
    x_sm, y_sm = smo.fit_resample(x, y)
    #affectation de x et y
    x = x_sm
    y = y_sm
    
    #afficher la répartition des valeurs dans la cible:
    st.markdown("Affichons la répartition des valeurs dans les catégories de la variable cible:")
    fig = plt.figure()
    sns.countplot(data = df, x = 'RainTomorrow_encode')
    st.pyplot(fig)
    
    if choice == 'Undersampling':
      st.markdown("Les données sont déséquilibrées")
    elif choice == 'OverSampling SMOTE':
      st.markdown("Les données sont déséquilibrées")
    elif choice == 'Aucun rééchantillonage':
      st.markdown("Les données sont déséquilibrées")  
  
  choice2 = st.selectbox('Select the items you want?', ('Aucune normalisation','StandardScaler'))
  #displaying the selected option
  st.write('You have selected:', choice2)

  if choice2 == 'StandardScaler':
    #affectation de x et y

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    name_columns_numerics = x.select_dtypes(include=numerics).columns
    #créer, Entrainer et transformer directement les colonnes numériques de x
    scaler =  StandardScaler()
    x[name_columns_numerics] = scaler.fit_transform(x[name_columns_numerics])
    
  #elif choice2 == 'Aucune normalisation':
    #affectation de x et y

  
  #potiron
  
    
if rad == "Machine Learning":
    
    #le train set
    #reformatage des dimensions de y pour permettre de rentrer les données dans traintestsplit :
    y = np.array(y)
    y.reshape(-1, 1)
    #le split
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        
    
    #selection du modèle
    choice3 = st.selectbox('Selecttionez les modèle :',('KNN','arbre de décision','régression logistique','Random forest'))

    if choice3 == 'KNN':
      model = KNeighborsClassifier(metric='manhattan', n_neighbors=26, weights='distance')
      
      #itération du modèle :
      st.markdown('itération du modèle')
      model.fit(x_train,y_train)
      ##Précision et f1-score :
      y_pred_train = model.predict(x_train)
      y_pred_test = model.predict(x_test)
      
    elif choice3 == 'arbre de décision' :
      model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7, min_samples_leaf = 40, random_state = 123)

      #itération du modèle :
      st.markdown('itération du modèle')
      model.fit(x_train,y_train)
      ##Précision et f1-score :
      y_pred_train = model.predict(x_train)
      y_pred_test = model.predict(x_test)
      
    elif choice3 == 'régression logistique' :
      model = LogisticRegression(C=0.01, penalty= 'l2')

      #itération du modèle :
      st.markdown('itération du modèle')
      model.fit(x_train,y_train)
      ##Précision et f1-score :
      y_pred_train = model.predict(x_train)
      y_pred_test = model.predict(x_test)
      
    elif choice3 == 'Random forest' :
      model = RandomForestClassifier(max_depth = 8, n_estimators = 200, criterion = 'gini', max_features = 'sqrt')
    
      #itération du modèle :
      st.markdown('itération du modèle')
      model.fit(x_train,y_train)
      ##Précision et f1-score :
      y_pred_train = model.predict(x_train)
      y_pred_test = model.predict(x_test)
    
    
   
    st.markdown('Maintenant que le modèle est entrainé, voyons la qualité de la prédiction')
    
    choice4 = st.selectbox('Choisissez une métrique ?',('accuracy','F1-score','matrice de confusion','AUC et ROC Curve','MAE'))

    if choice4 == 'accuracy':
      acc_train  = accuracy_score(y_train, y_pred_train)
      acc_test  = accuracy_score(y_test, y_pred_test)
      st.write("acc_train : ", acc_train, "acc_test :", acc_test)
      
    elif choice4 == 'F1-score' :
      f1score_train = f1_score(y_train, y_pred_train, average='macro')
      f1score_test = f1_score(y_test, y_pred_test, average='macro')
      st.write("F1score_train : ", f1score_train, "F1score_test : ", f1score_test)

    elif choice4 == 'matrice de confusion' :
      st.write(pd.crosstab(y_sm_test, y_pred_test, rownames=['Classe réelle'], colnames=['Classe prédite']))

    elif choice4 == 'AUC et ROC Curve' :
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
      st.write('Le score AUC est de', roc_auc_score, 'interprétation : plus il est proche de 1 plus le modèle est précis, plus il est proche 0.5 moins le modèle est précis.');
      st.markdown('Le score AUC ici est donc acceptable. ')
      st.markdown("Le classement des vrais positifs est cependant moins bon que le classement des vrais négatifs")
      st.markdown('Les scores d accuracy (précision globale) et de f1-score (sensible à la précision de prédiction de chaque classe) sur les jeux d entrainement et de test sont :  ')

    elif choice4 == 'MAE' :
      MAE = mae(y_test, y_pred_test)
      st.write("La 'Mean Absolute Error' ou 'MAE' est de : " + str(MAE), ', plus elle est basse plus le modèle est précis. Notre modèle a donc ici une précision correcte, ce paramètre d erreur est cohérent et confirme le score de précision. ')

      
    
    #résultats :
    st.markdown("Les prédictions sont plutôt bonnes !")
    st.markdown("Il y a un meilleur classement des positifs (classe 1). Le f1-score est correct également.")
    

if rad == "Conclusion et perspectives":
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










