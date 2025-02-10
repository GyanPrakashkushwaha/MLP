import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------------------------------
# DATA GENERATION
# ------------------------------------------------------------------------------------
st.sidebar.header("Synthetic Data Settings")
n_samples = st.sidebar.slider("Number of Samples", min_value=20, max_value=200, value=50, step=10)

# Generate synthetic data for three features:
# CGPA: 6 to 10, Age: 20 to 30, Projects: 1 to 5.
CGPA = np.random.uniform(6, 10, n_samples)
Age = np.random.uniform(20, 30, n_samples)
Projects = np.random.randint(1, 6, n_samples)

# For simplicity, assume the true underlying relationship is linear:
# Salary = 1000*CGPA + 500*Age + 300*Projects + noise
noise = np.random.normal(0, 500, n_samples)
Salary = 1000 * CGPA + 500 * Age + 300 * Projects + noise

# Create a DataFrame for easy viewing
data = pd.DataFrame({
    'CGPA': CGPA,
    'Age': Age,
    'Projects': Projects,
    'Salary': Salary
})

st.title("Visualization Tool: Regression Models")
st.subheader("Synthetic Data Preview")
st.dataframe(data.head())

# ------------------------------------------------------------------------------------
# FUNCTIONS FOR MODEL FITTING
# ------------------------------------------------------------------------------------

def fit_multiple_linear_regression(X, y):
    """
    Fits a Multiple Linear Regression using the normal equation.
    """
    # Add a column of ones for the intercept
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    # Normal Equation: beta = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
    return beta

def predict_multiple_linear_regression(X, beta):
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_design @ beta

def fit_polynomial_regression(X, y, degree=2):
    """
    Fits a Polynomial Regression model by expanding the features and then applying the normal equation.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    beta = np.linalg.inv(X_poly.T @ X_poly) @ (X_poly.T @ y)
    return beta, poly

def predict_polynomial_regression(X, beta, poly):
    X_poly = poly.transform(X)
    return X_poly @ beta

def gaussian_kernel(x, xi, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel between x and xi.
    """
    return np.exp(-np.linalg.norm(x - xi)**2 / (2 * sigma**2))

def predict_kernel_regression(X_train, y_train, x_query, sigma=1.0):
    """
    Predicts the target value for x_query using the Nadaraya-Watson Kernel Regression.
    """
    # Compute weights for each training point
    weights = np.array([gaussian_kernel(x_query, xi, sigma) for xi in X_train])
    if np.sum(weights) == 0:
        return 0
    return np.sum(weights * y_train) / np.sum(weights)

def batch_predict_kernel_regression(X_train, y_train, X_query, sigma=1.0):
    """
    Batch prediction for kernel regression.
    """
    predictions = np.array([predict_kernel_regression(X_train, y_train, xq, sigma) for xq in X_query])
    return predictions

# ------------------------------------------------------------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------------------------------------------------------------
st.sidebar.title("Select Regression Model")
model_choice = st.sidebar.radio("Choose a model", ("Multiple Linear Regression",
                                                   "Polynomial Regression (Degree 2)",
                                                   "Kernel Regression (Gaussian)"))

# Feature matrix and target vector
X = data[['CGPA', 'Age', 'Projects']].values
y = data['Salary'].values

# Create an area to display mathematical formulations
st.markdown("## Mathematical Formulations")

if model_choice == "Multiple Linear Regression":
    st.markdown("### Multiple Linear Regression (MLR)")
    st.latex(r"""
    y = \beta_0 + \beta_1 \cdot \text{CGPA} + \beta_2 \cdot \text{Age} + \beta_3 \cdot \text{Projects} + \varepsilon
    """)
    st.markdown("**Matrix Form:**")
    st.latex(r"""
    \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
    """)
    st.markdown("**Normal Equation:**")
    st.latex(r"""
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
    """)
    
    # Fit the Multiple Linear Regression model
    beta_mlr = fit_multiple_linear_regression(X, y)
    y_pred = predict_multiple_linear_regression(X, beta_mlr)
    
    st.subheader("Model Coefficients")
    st.write(f"Intercept (β₀): {beta_mlr[0]:.2f}")
    st.write(f"CGPA (β₁): {beta_mlr[1]:.2f}")
    st.write(f"Age (β₂): {beta_mlr[2]:.2f}")
    st.write(f"Projects (β₃): {beta_mlr[3]:.2f}")
    
    
    # Demonstration with first 5 data points
    st.subheader("Example Computation with First 5 Data Points")
    X_first5 = X[:5]
    y_first5 = y[:5]
    X_design_first5 = np.hstack([np.ones((5, 1)), X_first5])
    XtX = X_design_first5.T @ X_design_first5
    Xty = X_design_first5.T @ y_first5
    beta_first5 = np.linalg.inv(XtX) @ Xty

    st.write("**Design Matrix (X) for first 5 points:**")
    st.dataframe(pd.DataFrame(X_design_first5, columns=['Intercept', 'CGPA', 'Age', 'Projects']))
    st.write("**XᵀX Matrix:**")
    st.write(pd.DataFrame(XtX))
    st.write("**Xᵀy Vector:**")
    st.write(pd.DataFrame(Xty))
    st.write("**Computed Coefficients (β) using first 5 points:**")
    st.write(pd.DataFrame(beta_first5, index=['Intercept', 'CGPA', 'Age', 'Projects'], columns=['Coefficient']))

elif model_choice == "Polynomial Regression (Degree 2)":
    st.markdown("### Polynomial Regression (Degree 2)")
    st.latex(r"""
    \begin{aligned}
    y =\,& \beta_0 + \beta_1 \cdot \text{CGPA} + \beta_2 \cdot \text{Age} + \beta_3 \cdot \text{Projects} \\
    &+ \beta_4 \cdot \text{CGPA}^2 + \beta_5 \cdot \text{Age}^2 + \beta_6 \cdot \text{Projects}^2 \\
    &+ \beta_7 \cdot \text{CGPA}\cdot\text{Age} + \beta_8 \cdot \text{CGPA}\cdot\text{Projects} + \beta_9 \cdot \text{Age}\cdot\text{Projects} + \varepsilon
    \end{aligned}
    """)
    st.markdown("**Matrix Form (after expansion):**")
    st.latex(r"""
    \mathbf{y} = \mathbf{X}_{\text{poly}}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
    """)
    st.markdown("**Normal Equation:**")
    st.latex(r"""
    \hat{\boldsymbol{\beta}} = (\mathbf{X}_{\text{poly}}^T \mathbf{X}_{\text{poly}})^{-1}\mathbf{X}_{\text{poly}}^T\mathbf{y}
    """)
    
    # Fit the Polynomial Regression model (degree 2)
    beta_poly, poly_transformer = fit_polynomial_regression(X, y, degree=2)
    y_pred = predict_polynomial_regression(X, beta_poly, poly_transformer)
    
    st.subheader("Model Coefficients (First 10)")
    # Display only the first 10 coefficients (there can be many features after expansion)
    for i, coef in enumerate(beta_poly):
        st.write(f"β_{i}: {coef:.2f}")
        
      # Show polynomial features for first 5 points
    st.subheader("Polynomial Feature Expansion for First 5 Points")
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly_first5 = poly.fit_transform(X[:5])
    feature_names = poly.get_feature_names_out(['CGPA', 'Age', 'Projects'])
    st.dataframe(pd.DataFrame(X_poly_first5, columns=feature_names))

elif model_choice == "Kernel Regression (Gaussian)":
    st.markdown("### Kernel Regression (Nadaraya–Watson Estimator)")
    st.latex(r"""
    \hat{y}(X) = \frac{\sum_{i=1}^{n} K(X, X_i) \, y_i}{\sum_{i=1}^{n} K(X, X_i)}
    """)
    st.markdown("**Gaussian Kernel:**")
    st.latex(r"""
    K(X, X_i) = \exp\left(-\frac{\|X - X_i\|^2}{2\sigma^2}\right)
    """)
    
    sigma = st.sidebar.slider("Kernel Sigma (Bandwidth)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    # For kernel regression, we will predict on the training set for visualization.
    y_pred = batch_predict_kernel_regression(X, y, X, sigma=sigma)
    
    st.subheader("Kernel Weights for First 5 Data Points")
    X_first5 = X[:5]
    for i, xq in enumerate(X_first5):
        st.write(f"#### Query Point {i+1} (CGPA={xq[0]:.2f}, Age={xq[1]:.2f}, Projects={xq[2]:.1f})")
        weights = np.array([gaussian_kernel(xq, xi, sigma) for xi in X])
        df_weights = pd.DataFrame({
            'Training Point': np.arange(n_samples),
            'Weight': weights
        })
        st.dataframe(df_weights.sort_values(by='Weight', ascending=False).head(10))  # Show top 10 weights
        st.write(f"Predicted Salary: {np.sum(weights * y) / np.sum(weights):.2f}")
    
# ------------------------------------------------------------------------------------
# PLOTTING: True vs. Predicted Salary
# ------------------------------------------------------------------------------------
st.markdown("## Model Performance Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y, y_pred, color="blue", alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("True Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title(f"True vs. Predicted Salary: {model_choice}")
st.pyplot(fig)

# ------------------------------------------------------------------------------------
# DISPLAY COMPLETE MATHEMATICAL DETAILS
# ------------------------------------------------------------------------------------
st.markdown("## Complete Mathematical Explanation")

st.markdown("### 1️⃣ Multiple Linear Regression (MLR)")
st.latex(r"""
y = \beta_0 + \beta_1 \cdot \text{CGPA} + \beta_2 \cdot \text{Age} + \beta_3 \cdot \text{Projects} + \varepsilon
""")
st.markdown("**Matrix Form:**")
st.latex(r"""
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
""")
st.markdown("**Normal Equation:**")
st.latex(r"""
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
""")

st.markdown("---")

st.markdown("### 2️⃣ Polynomial Regression (Degree 2)")
st.latex(r"""
\begin{aligned}
y =\,& \beta_0 + \beta_1 \cdot \text{CGPA} + \beta_2 \cdot \text{Age} + \beta_3 \cdot \text{Projects} \\
&+ \beta_4 \cdot \text{CGPA}^2 + \beta_5 \cdot \text{Age}^2 + \beta_6 \cdot \text{Projects}^2 \\
&+ \beta_7 \cdot \text{CGPA}\cdot\text{Age} + \beta_8 \cdot \text{CGPA}\cdot\text{Projects} + \beta_9 \cdot \text{Age}\cdot\text{Projects} + \varepsilon
\end{aligned}
""")
st.markdown("**Matrix Form (after expansion):**")
st.latex(r"""
\mathbf{y} = \mathbf{X}_{\text{poly}}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
""")
st.markdown("**Normal Equation:**")
st.latex(r"""
\hat{\boldsymbol{\beta}} = (\mathbf{X}_{\text{poly}}^T \mathbf{X}_{\text{poly}})^{-1}\mathbf{X}_{\text{poly}}^T\mathbf{y}
""")

st.markdown("---")

st.markdown("### 3️⃣ Kernel Regression (Nadaraya–Watson)")
st.latex(r"""
\hat{y}(X) = \frac{\sum_{i=1}^{n} K(X, X_i) \, y_i}{\sum_{i=1}^{n} K(X, X_i)}
""")
st.markdown("**Gaussian Kernel:**")
st.latex(r"""
K(X, X_i) = \exp\left(-\frac{\|X - X_i\|^2}{2\sigma^2}\right)
""")