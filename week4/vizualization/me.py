import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
import time

# Title of the app
st.title("Understanding Stochastic Gradient Descent (SGD) Regression")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
n_features = st.sidebar.slider("Number of features", 1, 10, 2)
noise = st.sidebar.slider("Noise level", 1, 100, 10)
learning_rate = st.sidebar.slider("Learning rate", 0.001, 1.0, 0.01)
max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 500)

# Generate synthetic regression data
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)

# Play/Pause button
play = st.sidebar.checkbox("Play/Pause Animation")

# Initialize model
sgd = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=1, warm_start=True, random_state=42)

# Visualization
st.subheader("Regression Visualization with Iterative Learning")
fig, ax = plt.subplots()

if n_features > 1:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', label="Data points")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Target (y)")
else:
    ax.scatter(X, y, color="blue", label="Actual Data")
    line, = ax.plot([], [], color="red", label="SGD Regression Line")
    ax.legend()
    st.pyplot(fig)
    
    for i in range(1, max_iter + 1):
        if not play:
            break
        sgd.partial_fit(X, y)
        y_pred = sgd.predict(X)
        line.set_data(X.flatten(), y_pred)
        ax.relim()
        ax.autoscale_view()
        st.pyplot(fig, clear_figure=True)
        time.sleep(0.2)

# Display model coefficients
st.subheader("Model Coefficients")
# st.write(f"Intercept (θ₀): {sgd.intercept_[0]:.4f}")
# st.write(f"Coefficients (θ₁ to θₙ): {sgd.coef_}")

# Explanation Section
st.subheader("Understanding SGD Regression")
st.write("""
SGD Regression is an iterative optimization algorithm that updates model parameters using small subsets of the dataset. Unlike batch gradient descent, which uses the entire dataset for each update, SGD updates parameters using one sample at a time, making it efficient for large datasets.
""")

st.subheader("Mathematical Explanation")
st.markdown("### 1. Hypothesis Function")
st.latex(r"h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n")

st.markdown("### 2. Cost Function (Mean Squared Error)")
st.latex(r"J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2")

st.markdown("### 3. Gradient Descent Update Rule")
st.latex(r"\theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}")

st.markdown("### 4. Parameter Update in SGD")
st.latex(r"\theta_j = \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}")

# Advantages and Disadvantages
st.subheader("Pros and Cons of SGD Regression")
st.markdown("""
**Advantages:**
- Works well for large datasets.
- Faster than traditional gradient descent.
- Can be used for online learning (continuous updates).

**Disadvantages:**
- Sensitive to feature scaling.
- Learning rate tuning is required.
- Can be unstable if the learning rate is too high.
""")

st.subheader("Final Thoughts")
st.write("""
SGD Regression is a powerful tool for large-scale machine learning problems. Understanding its workings and properly tuning its parameters can significantly improve model performance.
""")

if __name__ == "__main__":
    st.write("App is running!")
