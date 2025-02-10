import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to calculate cost
def cost_function(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Function to perform SGD manually
def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, epochs=100):
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(0, m)
            xi = X[rand_index, :].reshape(1, -1)
            yi = y[rand_index]
            prediction = xi.dot(theta)
            gradient = xi.T * (prediction - yi)
            theta = theta - learning_rate * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

# Streamlit App
st.title("SGD Regression Explained")
st.markdown("""
### What is SGD Regression?
Stochastic Gradient Descent (SGD) Regression is an iterative optimization technique used to minimize the cost function in linear regression.
It updates the model weights step by step using a randomly chosen data point.

### Mathematical Formulation
For a given dataset with features $X$ and target $y$, we want to find $\theta$ such that:
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 $$
where $h_{\theta}(x) = X\theta$ is the prediction function.

The weight update rule in SGD is:
$$ \theta := \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta} $$
where $\alpha$ is the learning rate.
""")

# Generate Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # Adding bias term

theta_initial = np.random.randn(2, 1)
learning_rate = st.slider("Select Learning Rate", 0.001, 0.1, 0.01)
epochs = st.slider("Select Epochs", 10, 500, 100)

# Perform SGD manually
theta_sgd, cost_history = stochastic_gradient_descent(X_b, y, theta_initial, learning_rate, epochs)

# Train using Scikit-learn's SGDRegressor
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
sgd_reg = SGDRegressor(max_iter=epochs, eta0=learning_rate, learning_rate='constant')
sgd_reg.fit(X_train_scaled, y_train.ravel())
y_pred = sgd_reg.predict(X_test_scaled)

# Plot cost history
st.write("### Cost Function Over Iterations")
fig, ax = plt.subplots()
ax.plot(range(len(cost_history)), cost_history, label='Cost Function')
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.legend()
st.pyplot(fig)

# Display results
st.write("### Final Parameters (Theta)")
st.write(f"Manually Computed Theta: {theta_sgd.ravel()}")
st.write(f"Scikit-learn SGDRegressor Coefficients: {sgd_reg.coef_}, Intercept: {sgd_reg.intercept_}")

st.write("### Model Performance")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")