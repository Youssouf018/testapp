import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
def main():
    st.title("Prédiction du prix d'une maison avec un Perceptron multicouche")
    st.subheader("Auteur : Youssouf")
    data=pd.read_csv('/content/Drive/MyDrive/house_price/data.csv')

    st.sidebar.checkbox('Afficher les données brutes', False)

    numerical_df = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Numerical Features Only)')
    plt.show()

    city_price = data.groupby('city')['price'].mean().sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    city_price.plot(kind='bar')
    plt.title('Average House Price per City')
    plt.xlabel('City')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.show()


    le = LabelEncoder()

    data['city'] = le.fit_transform(data['city'])
    data = data[(data['price'] >= 50000) & (data['price'] <= 1000000)]


    plt.figure(figsize=(10,5))
    sns.boxplot(x=data['price'])
    plt.title('Boxplot of House Prices')
    plt.show()

    data = data.drop(['sqft_lot', 'waterfront', 'condition', 'yr_built', 'yr_renovated', 'date', 'country', 'statezip', 'street'], axis=1)

    #prétraitement
    X=data.drop('price', axis=1)
    y=data['price']
    data.shape
    print(X.shape)
    print(y.shape)
    (X == 0).any()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.sidebar.button("Exécuter", key="Classify"):
        st.header("Résultats du modèle")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)


    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    st.subheader("----- Random Forest Results -----")
    st.write(f"MSE: {mse_rf}")
    st.write(f"RMSE: {rmse_rf}")
    st.write(f"R2 Score: {r2_rf}")

    y_test_sorted = np.array(y_test).flatten()
    y_pred_rf_sorted = np.array(y_pred_rf).flatten()


    sorted_indices = np.argsort(y_test_sorted)
    y_test_sorted = y_test_sorted[sorted_indices]
    y_pred_rf_sorted = y_pred_rf_sorted[sorted_indices]

    window_size = 30  
    y_pred_rf_smoothed = pd.Series(y_pred_rf_sorted).rolling(window=window_size, center=True, min_periods=1).mean()

    plt.figure(figsize=(10,6))
    plt.plot(y_test_sorted, label='Actual Prices', color='blue')
    plt.plot(y_pred_rf_smoothed, label='Predicted Prices (Smoothed)', color='orange')

    plt.xlabel('Sample Index')
    plt.ylabel('House Price')
    plt.title('Actual vs Predicted Prices (Random Forest, Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

