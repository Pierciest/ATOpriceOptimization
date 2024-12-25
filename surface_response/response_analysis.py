import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

def fit_and_visualize_surface(revenue_df, results_path="results"):
    # Add polynomial features for metamodeling
    revenue_df['P1^2'] = revenue_df['P1'] ** 2
    revenue_df['P2^2'] = revenue_df['P2'] ** 2
    revenue_df['P1_P2'] = revenue_df['P1'] * revenue_df['P2']

    # Normalize features and target
    X = revenue_df[['P1', 'P2', 'P1^2', 'P2^2', 'P1_P2']].values
    y = revenue_df['Revenue'].values
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Fit Ridge regression
    model = Ridge(alpha=1e-3)
    model.fit(X_normalized, y_normalized)

    # Define response function
    intercept = model.intercept_
    slope = model.coef_
    response_function = lambda s, d: intercept + slope[0]*s + slope[1]*d + slope[2]*s*d + slope[3]*s**2 + slope[4]*d**2

    # Visualize the response surface
    x1_vals = np.linspace(-1, 1, 100)
    x2_vals = np.linspace(-1, 1, 100)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    z_vals = response_function(x1_grid, x2_grid)
    z_vals_original = scaler_y.inverse_transform(z_vals.reshape(-1, 1)).reshape(z_vals.shape)

    # 3D Surface plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_vals_original, cmap='viridis', edgecolor='none')
    ax.set_title("Response Surface")
    ax.set_xlabel("P1")
    ax.set_ylabel("P2")
    ax.set_zlabel("Revenue")
    plt.savefig(f"{results_path}/response_surface.png")
    plt.close()

    # Contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x1_grid, x2_grid, z_vals_original, cmap='viridis', levels=50)
    plt.colorbar(contour)
    plt.title("Contour Plot of the Revenue Surface")
    plt.xlabel("P1")
    plt.ylabel("P2")
    plt.savefig(f"{results_path}/response_contour.png")
    plt.close()

    return model, scaler_X, scaler_y, response_function
