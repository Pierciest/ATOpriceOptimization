import matplotlib.pyplot as plt
import numpy as np
def plot_revenue_vs_scenarios(scenario_steps, revenue_results):
    """
    Plot net revenue against the number of scenarios.
    Args:
        scenario_steps (list): Number of scenarios.
        revenue_results (dict): Revenue results for different price sets.
    """
    plt.figure(figsize=(10, 6))
    for price_set, revenues in revenue_results.items():
        plt.plot(scenario_steps, revenues, label=f"Prices: {np.round(price_set, 2)}")
    plt.title("Net Revenue vs Number of Scenarios")
    plt.xlabel("Number of Scenarios")
    plt.ylabel("Net Revenue")
    plt.legend()
    plt.grid()
    plt.show()
