import numpy as np
import matplotlib.pyplot as plt
from config.parameters import NUM_PRODUCTS, NUM_SCENARIOS

def generate_price_sets(costs, n_sets=10):
    """
    Generate random price sets based on costs.
    Args:
        costs (list): List of costs for each product.
        n_sets (int): Number of unique price sets to generate.

    Returns:
        list: List of price sets, each being a list of prices.
    """
    return [
        [np.random.uniform(cost, cost + 1000) for cost in costs]
        for _ in range(n_sets)
    ]

def generate_demand_scenarios(price_sets, num_scenarios, num_products, noise_std=10):
    """
    Generate demand scenarios using price and cross-price elasticity.
    Args:
        price_sets (list): List of price sets (one set for each scenario).
        num_scenarios (int): Number of scenarios to generate.
        num_products (int): Number of products.
        noise_std (float): Standard deviation of noise.

    Returns:
        numpy.ndarray: Demand scenarios for all products and scenarios.
    """
    num_price_sets = len(price_sets)
    demands = np.zeros((num_price_sets, num_scenarios, num_products))

    for price_set_idx, price_set in enumerate(price_sets):
        for scenario in range(num_scenarios):
            for product_idx in range(num_products):
                alpha = np.random.uniform(50, 200)  # Baseline demand for product
                beta = np.random.uniform(0.5, 5.0)  # Own-price elasticity
                noise = np.random.normal(0, noise_std)  # Random noise

                # Cross-price effects
                cross_price_effect = 0
                for other_product_idx in range(num_products):
                    if other_product_idx != product_idx:
                        gamma = np.random.uniform(-1.0, 1.0)  # Cross-price elasticity
                        cross_price_effect += gamma * price_set[other_product_idx]

                # Demand formula
                price = price_set[product_idx]
                demand = max(0, alpha - beta * price + cross_price_effect + noise)
                demands[price_set_idx, scenario, product_idx] = demand

    return demands

def simulate_demand_scenarios(gozinto_matrix, component_costs, price_sets, scenario_steps, num_products, noise_std=10):
    """
    Simulate revenue over multiple demand scenarios.
    Args:
        gozinto_matrix (numpy.ndarray): Gozinto matrix.
        component_costs (numpy.ndarray): Component costs.
        price_sets (list): List of price sets.
        scenario_steps (list): Number of demand scenarios to simulate.
        num_products (int): Number of products.
        noise_std (float): Standard deviation of noise for demand generation.

    Returns:
        dict: Dictionary of revenues over scenarios for each price set.
    """
    revenues_over_scenarios = {tuple(prices): [] for prices in price_sets}

    for num_scenarios in scenario_steps:
        # Generate demand scenarios for all price sets
        demand_scenarios = generate_demand_scenarios(price_sets, num_scenarios, num_products, noise_std)

        for price_set_idx, price_set in enumerate(price_sets):
            total_revenue = 0

            for demand in demand_scenarios[price_set_idx]:  # Iterate over scenarios for the current price set
                # Compute component usage and costs
                component_usage = gozinto_matrix @ demand
                total_costs = np.sum(component_costs * component_usage)

                # Compute revenue for the scenario
                scenario_revenue = np.sum(np.array(price_set) * demand) - total_costs
                total_revenue += scenario_revenue

            # Store average revenue
            revenues_over_scenarios[tuple(price_set)].append(total_revenue / num_scenarios)

    return revenues_over_scenarios

def plot_revenues_over_scenarios(revenues_over_scenarios, scenario_steps):
    """
    Plot revenue stability over increasing demand scenarios.
    Args:
        revenues_over_scenarios (dict): Dictionary of revenues for each price set.
        scenario_steps (list): Number of scenarios simulated.

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))
    for price_set, revenues in revenues_over_scenarios.items():
        plt.plot(scenario_steps, revenues, label=f"Prices: {np.round(price_set, 2)}")

    plt.title("Net Revenue vs Number of Scenarios")
    plt.xlabel("Number of Scenarios")
    plt.ylabel("Net Revenue")
    plt.legend()
    plt.grid()
    plt.show()

def run_simulation(gozinto_matrix, component_costs, costs, scenario_steps, num_products, n_price_sets=10):
    """
    Run the complete simulation and plot results.
    Args:
        gozinto_matrix (numpy.ndarray): Gozinto matrix.
        component_costs (numpy.ndarray): Component costs.
        costs (list): List of product costs.
        scenario_steps (list): Number of demand scenarios to simulate.
        num_products (int): Number of products.
        n_price_sets (int): Number of price sets to generate.

    Returns:
        None
    """
    # Step 1: Generate price sets
    price_sets = generate_price_sets(costs, n_sets=n_price_sets)

    # Step 2: Simulate demand and revenue
    revenues_over_scenarios = simulate_demand_scenarios(
        gozinto_matrix, component_costs, price_sets, scenario_steps, num_products
    )

    # Step 3: Plot results
    plot_revenues_over_scenarios(revenues_over_scenarios, scenario_steps)
