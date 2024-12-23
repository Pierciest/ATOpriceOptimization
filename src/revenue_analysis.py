import numpy as np

def evaluate_revenue(price_set, demand_scenarios, gozinto_matrix, component_costs):
    """
    Evaluate net revenue for given price sets and demand scenarios.
    Args:
        price_set (list): Product prices.
        demand_scenarios (np.ndarray): Simulated demand scenarios.
        gozinto_matrix (np.ndarray): Gozinto factor matrix.
        component_costs (np.ndarray): Costs of components.

    Returns:
        float: Average net revenue.
    """
    total_revenue = 0
    for demand in demand_scenarios:
        # Component usage and costs
        component_usage = gozinto_matrix @ demand
        total_cost = np.sum(component_costs * component_usage)

        # Revenue calculation
        revenue = np.sum(np.array(price_set) * demand) - total_cost
        total_revenue += revenue

    return total_revenue / len(demand_scenarios)
