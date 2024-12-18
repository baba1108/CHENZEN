import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import requests
import io
from io import StringIO  


#URL du fichier CSV
file_url ='https://drive.google.com/uc?export=download&id=1mFgByuwFgTUxGvfDtXGKaCi2oxfsiREw'
#telecharger le fichier
response = requests.get(file_url)
dataset = io.stringIO(response.text)


st.markdown("""
    <style>
        /* Couleur de fond de la page */
        body {
            background-color: #f0f8ff;  /* Bleu clair */
        }

        /* Couleur du texte principal */
        .css-1d391kg {
            color: #2e8b57;  /* Vert forêt */
        }

        /* Changer la couleur des boutons */
        .css-1emrehy.edgvbvh3 {
            background-color: #ff6347;  /* Tomate */
            color: white;
        }

        /* Personnalisation des titres */
        h1 {
            color: #ff1493;  /* Rose profond */
        }

        h2 {
            color: #4682b4;  /* Bleu acier */
        }

        h3 {
            color: #8a2be2;  /* Bleu violet */
        }

    </style>
""", unsafe_allow_html=True)

st.title('streamlit checkpoint1')
st.header(" Expresso Churn ")

st.sidebar.title("Options")
show_data = st.sidebar.checkbox("Afficher les données", value=True)
show_histograms = st.sidebar.checkbox("Afficher les histogrammes", value=True)
show_model_evaluation = st.sidebar.checkbox("Afficher l'évaluation du modèle", value=True)


# Lire le fichier CSV dans un DataFrame pandas
data = pd.read_csv('Expresso_churn_dataset.csv')
st.dataframe(data.head())


# Afficher les 5 premières lignes du DataFrame
st.write("Aperçu du Dataset :")
st.write("Informations du dataset :")
st.text(data.info())
st.write("Valeurs manquantes par colonne :")
st.write(data.isnull().sum())
print(data.head())
print(data.duplicated().sum())
print(data.describe())
#Fill missing numerical values with the mean
for col in data.select_dtypes(include=np.number):
  data[col] = data[col].fillna(data[col].mean())
# Fill missing categorical values with the most frequent value
for col in data.select_dtypes(include='object'):
    data[col] = data[col].fillna(data[col].mode()[0])


# Handling outliers (using IQR method for numerical features)
for col in data.select_dtypes(include=np.number):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Afficher des histogrammes des colonnes numériques avant et après le prétraitement
st.write("Histogrammes des caractéristiques numériques (Avant/Après prétraitement) :")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogramme avant prétraitement
data_original = pd.read_csv('Expresso_churn_dataset.csv')
data_original.select_dtypes(include=np.number).hist(ax=axes[0])
axes[0].set_title("Avant Prétraitement")

# Histogramme après prétraitement
data.select_dtypes(include=np.number).hist(ax=axes[1])
axes[1].set_title("Après Prétraitement")

st.pyplot(fig)

# Encoding categorical features
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Display the processed data
st.write("Données après encodage :")
st.dataframe(data.head())
print(data.info())


# Assuming 'churn' is your target variable
X = data.drop('CHURN', axis=1)
y = data['CHURN']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
st.write(f"Précision du modèle : {accuracy}")

# Rapport de classification
st.write("Rapport de classification :")
st.text(classification_report(y_test, y_pred))

# Matrice de confusion
st.write("Matrice de confusion :")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Churn", "Churn"], yticklabels=["Non Churn", "Churn"])
ax.set_xlabel('Prédit')
ax.set_ylabel('Réel')
st.pyplot(fig)

# Formulaire de saisie pour faire des prédictions
st.header("Faire une prédiction")

with st.form(key='prediction_form'):
    # Champs de saisie pour les fonctionnalités (en fonction de vos données)
    feature1 = st.number_input("Feature 1", min_value=0, max_value=1000, value=50)
    feature2 = st.number_input("Feature 2", min_value=0, max_value=1000, value=200)
    feature3 = st.number_input("Feature 3", min_value=0, max_value=1000, value=30)
    feature4 = st.number_input("Feature 4", min_value=0, max_value=1000, value=500)
    feature5 = st.number_input("Feature 5", min_value=0, max_value=1000, value=150)
