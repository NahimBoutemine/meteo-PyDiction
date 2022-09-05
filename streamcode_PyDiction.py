
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

rad = st.sidebar.radio("Menu",["Introduction : Le projet et ses créateurs", "Partie à resectionner"])

if rad == "Introduction : Le projet et ses créateurs":
  st.title("Introduction : Le projet et ses créateurs")
  
#Présentation du projet : titre, contexte, définition projet, objectifs du projet et de l'application:
  st.header("Le projet 'PyDiction'")
  if st.button('Say hello'):
     st.write('Why hello there')
    
  else:
    kangourou = Image.open('kangoufun.jpg')
    st.image(kangourou, caption=' ')
    st.markdown("Le titre du projet 'PyDiction' est une synthèse des mots prédiction et python car l'on cherche ici à prédire la pluie et python a été employé pour cela ainsi que le Machine Learning. ")
    st.markdown("Ce projet est réalisé dans le cadre d'une formation professionnelle en Data Science.")
    st.markdown("C'est un travail autour de la météorologie et du Machine Learning (ML). ")
    diapo = Image.open('diapo.jpg')
    st.image(diapo, caption="Etapes depuis les données météo jusqu'à la prédiction ")
    st.markdown("Les données météorologiques de pluie, ensoleillement, températures, pression, vent, humidité, sur plusieurs années, sur 49 stations réparties sur tout le territoire australien.")
    st.markdown("Le but a été de contruire un modèle de prédiction de la variable nommée 'RainTomorrow'. ")
    st.markdown("'RainTomorrow' représente la présence de pluie à jour J + 1 sur un point du territoire australien) qui vaut tout simplement 1 si la pluie est > 1mm, 0 sinon.")
    st.markdown("Cette application streamlit vise à montrer les étapes du travail jusqu'à la détermination du modèle idéal de prédiction, ainsi que la préparation optimale sur un dataset de ce type.")

#Présentation des créateurs :
  st.header("Les créateurs") 
  #Richard = Image.open('Richard.jpg')
  #Richard = Richard.resize((200, 200)) 
  #st.image(Richard, caption='Richard')
  Nahim = Image.open('Nahim.png')
  Nahim = Nahim.resize((200, 200))
  st.image(Nahim, caption='Nahim')
  

if rad == "Partie à resectionner":
  
   #Le jeu de données
  st.markdown("Les données sont présentes sur 49 stations, sur plusieurs années, et comprennent les informations de : ensoleillement, humidité, vitesse et sens du vent, quantité de nuages, températures minimales et maximales etc.")
  st.markdown("La pluie est considérée comme présente au jour J si elle est strictement supérieure à 1mm. ")
  #si le fichier est chargé, alors lancer le code seulement ensuite (condition nécessaire sinon le code se lance trop tôt et bloque):
  uploaded_file = st.file_uploader("cliquer sur 'Browse' pour charger vos données")
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("Affichons le nombre de lignes:")
    st.write(len(df))

    st.write(df)
    st.markdown("S'il existe des valeurs manquantes, elles sont à enlever pour le bon déroulement de la modélisation, de même que les doublons. ")
    st.markdown("Affichons le pourcentage de ces fameuses valeurs manquantes, et ce pour chacune des variables :")
    percent_missing_df = df.isnull().sum() * 100 / len(df)
    st.write(percent_missing_df)
    st.markdown("Ainsi, les valeurs manquantes sont enlevées car, même si en général elles sont remplacées par une valeur (imputation statistique), selon notre expérience dans ce cas de prédiction cela ne fait que rajouter du temps de calcul")
    df = df.dropna()
    df = df.drop_duplicates()
    st.markdown("Vérifiez par vous-même à présent le nombre de données manquantes : on peut voir à présent qu'il n'y en a plus!")
    percent_missing_df = df.isnull().sum() * 100 / len(df)
    st.write(percent_missing_df)
    st.markdown("Affichons le pourcentage de valeurs non en doublon pour vérifier qu'elles ont été supprimées:")
    percentage_dupli = df.duplicated(keep=False).value_counts(normalize=True) * 100
    st.write(percentage_dupli)
    st.markdown("Affichons de nouveau le nombre de lignes, ce nombre même réduit par rapport au départ (plus de 50% de suppression) nous a permis d'avoir un score de prédiction suffisant :")
    st.write(len(df))
    st.markdown("A présent, les données sont automatiquement encodées, date est transformée en Année, Mois et Jours")

    #afficher la répartition des valeurs dans la cible A FAIRE ET DECRIRE

    #encodage des données

    ##Traitement de la variable 'date' :

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
      st.markdown("Vérifions les encodages : c'est vérifié. ")
      st.write(df)

    #heatmap
    st.markdown("A présent, il faut sélectionner les variables explicatives pour la modélisation.")
    st.markdown("Pour cela, nous allons afficher la matrice des corrélations")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)
    st.markdown("Suppression des variables explicatives corréllées à moins de 5% à la cible selon le test de Pearson qui sont 'WindDir3pm','Temp9am','WindDir9am'") 
    df.drop(['WindDir3pm','Temp9am','WindDir9am'], axis = 1) 

    st.markdown("Le jeu de données est ensuite découpé en jeu de test et d entrainement à hauteur de 20% et 80% respectivement. Puis un rééchantillonnage SMOTE est appliqué puisque nous avons de meilleures performances avec. Cependant il est à noter que les méthodes de normalisation ou de réduction de dimensions n ont pas amené d améloration des résultats, nous ne les avons donc pas conservées. ")
    #(méthode courante d'évaluation):
    y = df['RainTomorrow_encode']
    x = df.drop('RainTomorrow_encode', axis = 1)

    #préparation de l'oversampling SMOTE
    smo = SMOTE()
    x_sm, y_sm = smo.fit_resample(x, y)

    # réduction du jeu de données, répartition en jeux train et test
    x_sm_train, x_sm_test, y_sm_train, y_sm_test = train_test_split(x_sm, y_sm, test_size=0.20, random_state=42)

    #itération du modèle : KNN, avec les données réduites.
    st.markdown('Le modèle optimal est selon notre expérience un KNN que nous avons optimisé par gridsearch. Evaluons à présent notre modèle :  ')
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=26, weights='distance') #mettre ici le meilleur nbr_voisins trouvé plus haut
    model.fit(x_sm_train,y_sm_train)

    ##Précision et f1-score :
    y_pred_train_KNNsm = model.predict(x_sm_train)
    y_pred_test_KNNsm = model.predict(x_sm_test)
    st.markdown('Les scores d accuracy (précision globale) et de f1-score (sensible à la précision de prédiction de chaque classe) sur les jeux d entrainement et de test sont :  ')

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
    st.markdown("Les prédictions sont plutôt bonnes !")
    st.markdown("Il y a un meilleur classement des positifs (classe 1). Le f1-score est correct également.")

    #AUC et ROC Curve:
    st.markdown('Imprimons à présent la courbe ROC de ce modèle : ')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_sm_test, model.predict(x_sm_test), pos_label = 1)
    roc_auc_score_KNNsm = roc_auc_score(y_sm_test, model.predict(x_sm_test))
    #la courbe ROC
    fig = plt.figure();
    plt.plot(false_positive_rate, true_positive_rate);
    plt.plot([0, 1], ls="--");
    plt.plot([0, 0], [1, 0] , c=".7"), 
    plt.plot([1, 1] , c=".7");
    plt.ylabel('True Positive Rate');
    plt.xlabel('False Positive Rate');
    st.pyplot(fig);
    st.write('Le score AUC est de', roc_auc_score_KNNsm, 'interprétation : plus il est proche de 1 plus le modèle est précis, plus il est proche 0.5 moins le modèle est précis.');
    st.markdown('Le score AUC ici est donc acceptable. ')
    st.markdown("Le classement des vrais positifs est cependant moins bon que le classement des vrais négatifs")

    # MAE :
    MAE_KNNsm = mae(y_sm_test, y_pred_test_KNNsm)
    st.write("La 'Mean Absolute Error' ou 'MAE' est de : " + str(MAE_KNNsm), ', plus elle est basse plus le modèle est précis. Notre modèle a donc ici une précision correcte, ce paramètre d erreur est cohérent et confirme le score de précision. ')

              #conf que c'est bien une prez de l'efficacité et evaluation de notre modèle, dans ce cas présenter tout en allant à essentiel, ajouter un réso de neurone pour comparer efficacité, des graphiques intéractifs sur ROC, les classes diparates et de la dataviz, des clics pour passer à l'étape suivante et une conclusion et limite de la suite des travaux.










