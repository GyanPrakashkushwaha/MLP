import dash
from dash import dcc, html, Input, Output, State, callback_context, exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA

# Initialize the app with a Bootstrap theme for styling.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout: a sidebar for controls and a main area for visualization and information.
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Controls", className="display-6"),
            html.Hr(),
            html.Div([
                html.Label("Number of Samples"),
                dcc.Slider(
                    id="n_samples",
                    min=100,
                    max=1000,
                    step=50,
                    value=500,
                    marks={i: str(i) for i in range(100, 1001, 200)},
                )
            ], className="mb-3"),
            html.Div([
                html.Label("Number of Features"),
                dcc.Slider(
                    id="n_features",
                    min=1,
                    max=10,
                    step=1,
                    value=1,  # For animation, use one feature
                    marks={i: str(i) for i in range(1, 11)},
                )
            ], className="mb-3"),
            html.Div([
                html.Label("Noise Level"),
                dcc.Slider(
                    id="noise",
                    min=1,
                    max=100,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(0, 101, 20)},
                )
            ], className="mb-3"),
            html.Div([
                html.Label("Learning Rate"),
                dcc.Slider(
                    id="learning_rate",
                    min=0.001,
                    max=1.0,
                    step=0.001,
                    value=0.01,
                    marks={
                        0.001: "0.001",
                        0.25: "0.25",
                        0.5: "0.5",
                        0.75: "0.75",
                        1.0: "1.0",
                    },
                )
            ], className="mb-3"),
            html.Div([
                html.Label("Maximum Iterations"),
                dcc.Slider(
                    id="max_iter",
                    min=100,
                    max=1000,
                    step=100,
                    value=500,
                    marks={i: str(i) for i in range(100, 1001, 200)},
                )
            ], className="mb-3"),
            html.Div([
                dbc.Checklist(
                    options=[{"label": "Play Animation (only for single feature)", "value": 1}],
                    value=[],  # When checked, value will contain 1.
                    id="play",
                    switch=True,
                )
            ], className="mb-3"),
            dbc.Button(
                "Reset Training",
                id="reset",
                color="primary",
                style={"display": "block", "width": "100%"},
            ),
        ], width=3),
        dbc.Col([
            html.H1("Understanding Stochastic Gradient Descent (SGD) Regression", className="display-4"),
            dcc.Graph(id="regression-graph"),
            # Display the current iteration count.
            html.Div(id="iteration-display", style={"marginBottom": "20px", "fontWeight": "bold"}),
            html.Hr(),
            html.H4("Model Coefficients"),
            html.Div(id="model-coefficients"),
            html.Hr(),
            html.H4("SGD Regression Explanation"),
            dcc.Markdown(
                """
**SGD Regression** is an iterative optimization algorithm that updates model parameters using small subsets of the data (even one sample at a time). Unlike batch gradient descent (which uses the entire dataset per update), SGD can be faster for large datasets and is also well-suited for online learning.

### Mathematical Overview
1. **Hypothesis Function:**  
   \\[
   h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + \\dots + \\theta_n x_n
   \\]

2. **Cost Function (Mean Squared Error):**  
   \\[
   J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2
   \\]

3. **Gradient Descent Update Rule:**  
   \\[
   \\theta_j = \\theta_j - \\alpha \\frac{\\partial J(\\theta)}{\\partial \\theta_j}
   \\]

4. **SGD Parameter Update (using one sample at a time):**  
   \\[
   \\theta_j = \\theta_j - \\alpha \\, (h_\\theta(x^{(i)}) - y^{(i)}) \\cdot x_j^{(i)}
   \\]

**Pros:**
- Efficient for large datasets.
- Faster convergence for online learning.
- Less memory usage.

**Cons:**
- Sensitive to feature scaling.
- Requires careful tuning of the learning rate.
- May be unstable with too high a learning rate.
                """
            ),
        ], width=9),
    ]),
    # Hidden stores for data and iteration counter.
    dcc.Store(id="data-store"),
    dcc.Store(id="iter-store", data=0),
    # Interval component to trigger iterative updates every 200ms (only enabled for single feature with play checked).
    dcc.Interval(id="interval-component", interval=200, n_intervals=0, disabled=True),
], fluid=True)

# Combined callback: update the generated data, iteration counter, and whether the interval is enabled.
@app.callback(
    Output("data-store", "data"),
    Output("iter-store", "data"),
    Output("interval-component", "disabled"),
    Input("n_samples", "value"),
    Input("n_features", "value"),
    Input("noise", "value"),
    Input("learning_rate", "value"),
    Input("max_iter", "value"),
    Input("reset", "n_clicks"),
    Input("play", "value"),
    Input("interval-component", "n_intervals"),
    State("data-store", "data"),
    State("iter-store", "data"),
)
def update_data_store(n_samples, n_features, noise, learning_rate, max_iter,
                      reset, play_value, n_intervals, current_data, current_iter):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Enable the interval only when play is checked and there is a single feature.
    interval_disabled = not (play_value and (1 in play_value) and n_features == 1)

    # When any parameter changes (or the reset button or play toggle), regenerate the data and reset iteration.
    if trigger_id in ["n_samples", "n_features", "noise", "learning_rate", "max_iter", "reset", "play"]:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
        new_data = {
            "X": X.tolist(),
            "y": y.tolist(),
            "n_features": n_features,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
        }
        new_iter = 0
        return new_data, new_iter, interval_disabled

    # Otherwise, if the interval fired, increment the iteration counter.
    elif trigger_id == "interval-component":
        if current_data is None:
            raise exceptions.PreventUpdate
        new_iter = current_iter if current_iter is not None else 0
        if new_iter < max_iter:
            new_iter += 1
        return current_data, new_iter, interval_disabled

    else:
        raise exceptions.PreventUpdate

# Callback to update the graph, model coefficients, and display the current iteration.
@app.callback(
    Output("regression-graph", "figure"),
    Output("model-coefficients", "children"),
    Output("iteration-display", "children"),
    Input("data-store", "data"),
    Input("iter-store", "data"),
)
def update_graph(data, iter_count):
    if data is None:
        return {}, "", ""
    
    # Convert stored lists back to numpy arrays.
    X = np.array(data["X"])
    y = np.array(data["y"])
    n_features = data["n_features"]
    learning_rate = data["learning_rate"]
    max_iter = data["max_iter"]

    # Create a new SGDRegressor instance.
    sgd = SGDRegressor(
        learning_rate="constant", 
        eta0=learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=42
    )

    if n_features == 1:
        # For one feature, call partial_fit at least once (even if iter_count is zero) to fit the model.
        for _ in range(max(1, iter_count)):
            sgd.partial_fit(X, y)
        y_pred = sgd.predict(X)

        # Create a scatter plot for the data and add the regression line.
        scatter = go.Scatter(
            x=X.flatten(),
            y=y,
            mode="markers",
            name="Actual Data",
            marker=dict(color="blue")
        )
        # Sort the x-values so the line is smooth.
        sort_idx = np.argsort(X.flatten())
        line = go.Scatter(
            x=X.flatten()[sort_idx],
            y=y_pred[sort_idx],
            mode="lines",
            name="SGD Regression Line",
            line=dict(color="red")
        )
        layout = go.Layout(
            title=f"SGD Regression (Iteration {iter_count} of {max_iter})",
            xaxis_title="Feature",
            yaxis_title="Target (y)",
            transition={"duration": 200},
        )
        fig = go.Figure(data=[scatter, line], layout=layout)
    else:
        # For multi-feature cases, reduce dimensions with PCA for visualization.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode="markers",
            marker=dict(color=y, colorscale="Portland", showscale=True),
            name="Data Points"
        )
        layout = go.Layout(
            title="PCA Projection of Data",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
        )
        fig = go.Figure(data=[scatter], layout=layout)
        # Fit the model once in multi-feature mode.
        sgd = SGDRegressor(learning_rate="constant", eta0=learning_rate, max_iter=1000, random_state=42)
        sgd.fit(X, y)

    # Prepare model coefficients display.
    coef_text = []
    if hasattr(sgd, "intercept_"):
        coef_text.append(html.P(f"Intercept (θ₀): {sgd.intercept_[0]:.4f}"))
    if hasattr(sgd, "coef_"):
        coef_text.append(html.P(f"Coefficients (θ₁ to θₙ): {np.array2string(sgd.coef_, precision=4)}"))

    # Display the current iteration.
    iter_text = f"Current Iteration: {iter_count}"

    return fig, coef_text, iter_text

if __name__ == "__main__":
    app.run_server(debug=True)
