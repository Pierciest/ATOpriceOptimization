import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def save_revenue_plots(revenue_df, results_path, scenario_revenues=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Dot plot for average revenues
    plt.figure(figsize=(10, 6))
    plt.scatter(revenue_df['P1'], revenue_df['Revenue'], label='P1 Revenue', color='blue', alpha=0.6, edgecolors='w', s=80)
    plt.scatter(revenue_df['P2'], revenue_df['Revenue'], label='P2 Revenue', color='red', alpha=0.6, edgecolors='w', s=80)
    plt.title("Revenue vs. Prices")
    plt.xlabel("Prices")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/dot_plot_revenues.png")
    plt.close()

    # Contour plot
    x = np.linspace(min(revenue_df['P1']), max(revenue_df['P1']), 100)
    y = np.linspace(min(revenue_df['P2']), max(revenue_df['P2']), 100)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.random.rand(100, 100)  # Placeholder for z values based on metamodel

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(x_grid, y_grid, z_grid, cmap='viridis', levels=50)
    plt.colorbar(contour)
    plt.title("Revenue Contour Plot")
    plt.xlabel("P1")
    plt.ylabel("P2")
    plt.savefig(f"{results_path}/revenue_contour.png")
    plt.close()

    # Scenario revenues plot (if provided)
    if scenario_revenues is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(scenario_revenues['Scenario'], scenario_revenues['Revenue'], label='Scenario Revenue', color='green', linestyle='-', marker='o')
        plt.title("Revenue vs. Scenarios")
        plt.xlabel("Scenario")
        plt.ylabel("Revenue")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_path}/scenario_vs_revenue.png")
        plt.close()
