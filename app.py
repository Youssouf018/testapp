import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def main():
    st.title("Prédiction du prix d'une maison avec un Perceptron multicouche")
    st.subheader("Auteur : Youssouf")

    # Chargement des données
    data = pd.read_csv("data.csv")

    if st.sidebar.checkbox('Afficher les données brutes', False):
        st.subheader("Jeu de données")
        st.write(data)

    seed = 123

    # Prétraitement
    dataset = data.select_dtypes(exclude="object")
    st.subheader("Données numériques (sans les variables catégorielles)")
    st.write(dataset)

    # Séparation X et y
    y = dataset["price"]
    X = dataset.drop("price", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    # Hyperparamètres
    st.sidebar.subheader("Hyper-paramètres du modèle")
    hidden_layer_sizes = st.sidebar.text_input("Taille des couches cachées ", value="100", max_chars=200)
    alpha = st.sidebar.number_input("Régularisation alpha", 0.0001, 1.0, value=0.0001, step=0.0001, format="%.4f")
    max_iter = st.sidebar.number_input("Itérations max", 100, 2000, value=1000, step=50)
    hidden_layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(","))


    if st.sidebar.button("Exécuter", key="Classify"):
        st.header("Résultats du modèle")


    # Entraînement
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        max_iter=max_iter,
        random_state=seed
    )
    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader(" Évaluation du modèle")
    st.write(f" MAE : {mae:.2f}")
    st.write(f" RMSE : {rmse:.2f}")

    # Formulaire pour prédiction personnalisée
    st.subheader(" Prédire le prix d'une maison personnalisée")
    user_input = {}
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

    if st.button("Prédire le prix"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f" Prix estimé : {prediction:,.2f} €")

if __name__ == '__main__':
    main()
