import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# Statistical Analysis of Revenue Trends
def analyze_revenue_stability(revenues_over_scenarios):
    """
    Calculate and display revenue stability metrics for each price set.
    Args:
        revenues_over_scenarios (dict): Dictionary of revenues for each price set.

    Returns:
        dict: Dictionary with variance and mean for each price set.
    """
    stability_metrics = {}
    for price_set, revenues in revenues_over_scenarios.items():
        variance = np.var(revenues)
        mean_revenue = np.mean(revenues)
        stability_metrics[price_set] = {
            "mean": mean_revenue,
            "variance": variance
        }

        print(f"Price Set: {np.round(price_set, 2)}")
        print(f"  Mean Revenue: {mean_revenue:.2f}")
        print(f"  Variance: {variance:.2f}\n")

    return stability_metrics

# Plotting Scatter of Price vs Demand for Each Scenario
def plot_price_vs_demand(price_sets, demand_scenarios):
    """
    Plot scatter graphs of price vs. demand for all scenarios.
    Args:
        price_sets (list): List of price sets.
        demand_scenarios (numpy.ndarray): Demand scenarios for all price sets and products.

    Returns:
        None
    """
    num_price_sets = len(price_sets)
    num_products = demand_scenarios.shape[2]

    for price_set_idx, price_set in enumerate(price_sets):
        for product_idx in range(num_products):
            plt.figure(figsize=(10, 6))
            for scenario in range(demand_scenarios.shape[1]):
                plt.scatter(price_set[product_idx], demand_scenarios[price_set_idx, scenario, product_idx], 
                            label=f"Scenario {scenario + 1}" if scenario < 10 else None, alpha=0.6)

            plt.title(f"Price vs Demand for Product {product_idx + 1} (Price Set {price_set_idx + 1})")
            plt.xlabel("Price")
            plt.ylabel("Demand")
            plt.grid()
            if scenario < 10:
                plt.legend()
            plt.show()

# Example Usage:
def analyze_and_plot(gozinto_matrix, component_costs, price_sets, scenario_steps, num_products, noise_std=10):
    """
    Perform additional analysis and visualization of revenue stability and price-demand relationship.
    Args:
        gozinto_matrix (numpy.ndarray): Gozinto matrix.
        component_costs (numpy.ndarray): Component costs.
        price_sets (list): List of price sets.
        scenario_steps (list): Number of demand scenarios to simulate.
        num_products (int): Number of products.
        noise_std (float): Standard deviation of noise for demand generation.

    Returns:
        None
    """
    from demand import simulate_demand_scenarios,generate_demand_scenarios

    # Simulate demand and revenue
    revenues_over_scenarios = simulate_demand_scenarios(
        gozinto_matrix, component_costs, price_sets, scenario_steps, num_products, noise_std
    )

    # Analyze revenue stability
    print("\nRevenue Stability Metrics:")
    stability_metrics = analyze_revenue_stability(revenues_over_scenarios)

    # Generate demand scenarios for all price sets
    num_scenarios = max(scenario_steps)
    demand_scenarios = generate_demand_scenarios(price_sets, num_scenarios, num_products, noise_std)

    # Plot price vs demand scatter
    print("\nPrice vs Demand Scatter Plots:")
    plot_price_vs_demand(price_sets, demand_scenarios)




