import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
import time

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "SGD Regression Visualization"

# Layout
app.layout = html.Div([
    html.H1("Understanding Stochastic Gradient Descent (SGD) Regression"),
    
    # Sidebar Controls
    html.Div([
        html.Label("Number of Samples"),
        dcc.Slider(id='n_samples', min=100, max=1000, step=50, value=500),
        
        html.Label("Number of Features"),
        dcc.Slider(id='n_features', min=1, max=10, step=1, value=2),
        
        html.Label("Noise Level"),
        dcc.Slider(id='noise', min=1, max=100, step=1, value=10),
        
        html.Label("Learning Rate"),
        dcc.Slider(id='learning_rate', min=0.001, max=1.0, step=0.01, value=0.01),
        
        html.Label("Maximum Iterations"),
        dcc.Slider(id='max_iter', min=100, max=1000, step=50, value=500),
        
        html.Button("Run SGD", id='run_sgd', n_clicks=0)
    ], style={'width': '30%', 'float': 'left', 'padding': '20px'}),
    
    # Graph Output
    html.Div([
        dcc.Graph(id='sgd-graph'),
        html.Div(id='coefficients-display')
    ], style={'width': '65%', 'float': 'right', 'padding': '20px'})
])

# Callback to update the graph
@app.callback(
    [Output('sgd-graph', 'figure'), Output('coefficients-display', 'children')],
    [Input('run_sgd', 'n_clicks')],
    [dash.State('n_samples', 'value'),
     dash.State('n_features', 'value'),
     dash.State('noise', 'value'),
     dash.State('learning_rate', 'value'),
     dash.State('max_iter', 'value')]
)
def update_graph(n_clicks, n_samples, n_features, noise, learning_rate, max_iter):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    
    sgd = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=1, warm_start=True, random_state=42)
    
    if n_features > 1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        for _ in range(max_iter):
            sgd.partial_fit(X, y)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                 mode='markers', marker=dict(color=y, colorscale='Viridis'),
                                 name='Data Points'))
        fig.update_layout(title='PCA Projection of Regression Data')
    else:
        for _ in range(max_iter):
            sgd.partial_fit(X, y)
        y_pred = sgd.predict(X)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines', name='SGD Regression Line', line=dict(color='red')))
        fig.update_layout(title='SGD Regression Visualization')
    
    coef_display = html.Div([
        html.H4("Model Coefficients"),
        html.P(f"Intercept (θ₀): {sgd.intercept_[0]:.4f}"),
        html.P(f"Coefficients (θ₁ to θₙ): {sgd.coef_}")
    ])
    
    return fig, coef_display

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)