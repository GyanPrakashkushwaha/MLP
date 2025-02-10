import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# Title and Description
st.title("Interactive Machine Learning Tutor")
st.write("Explore Linear Regression concepts with interactive exercises.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Select Topic", [
    "Introduction",
    "Dummy Regressor",
    "Linear Regression",
    "SGDRegressor",
    "Feature Scaling",
    "Model Evaluation",
    "Cross Validation"
])

if menu == "Introduction":
    st.header("Welcome to Interactive Machine Learning!")
    st.write("This app helps you understand Linear Regression with hands-on coding exercises.")
    st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png")

elif menu == "Dummy Regressor":
    st.header("Dummy Regressor - Baseline Model")
    strategy = st.selectbox("Select Strategy", ["mean", "median", "quantile", "constant"])
    X_train, X_test, y_train, y_test = train_test_split(np.random.rand(100, 1), np.random.rand(100), random_state=42)
    dummy_regr = DummyRegressor(strategy=strategy)
    dummy_regr.fit(X_train, y_train)
    score = dummy_regr.score(X_test, y_test)
    st.write(f"Dummy Regressor Score: {score:.4f}")

elif menu == "Linear Regression":
    st.header("Linear Regression Model")
    X, y = np.random.rand(100, 1) * 10, np.random.rand(100) * 50
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="Actual")
    ax.plot(X_test, y_pred, color='red', label="Predicted")
    ax.legend()
    st.pyplot(fig)

elif menu == "SGDRegressor":
    st.header("SGD Regressor - Stochastic Gradient Descent")
    X, y = np.random.rand(200, 1), np.random.rand(200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"SGDRegressor Score: {model.score(X_test, y_test):.4f}")

elif menu == "Feature Scaling":
    st.header("Feature Scaling for SGDRegressor")
    X = np.random.rand(100, 1) * 100  # Unscaled Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(X, ax=ax[0], kde=True)
    ax[0].set_title("Original Data")
    sns.histplot(X_scaled, ax=ax[1], kde=True)
    ax[1].set_title("Scaled Data")
    st.pyplot(fig)

elif menu == "Model Evaluation":
    st.header("Model Evaluation Metrics")
    X, y = np.random.rand(100, 1), np.random.rand(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared Score: {r2:.4f}")

elif menu == "Cross Validation":
    st.header("Cross Validation")
    X, y = np.random.rand(200, 1), np.random.rand(200)
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    st.write(f"Cross Validation Scores: {scores}")
    st.write(f"Mean Score: {np.mean(scores):.4f}")
