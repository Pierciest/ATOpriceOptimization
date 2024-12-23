import numpy as np
from config.parameters import PRICE_RANGE

def generate_price_demand_params():
    """
    Generate random parameters for price-demand relationships.
    Returns:
        tuple: (baseline demand, price sensitivity)
    """
    alpha = np.random.uniform(50, 200)  # Baseline demand
    beta = np.random.uniform(0.5, 5.0)  # Price sensitivity
    return alpha, beta

def simulate_demand(prices, n_samples=1000):
    """
    Simulate demand for products based on prices.
    Args:
        prices (list): List of product prices.
        n_samples (int): Number of samples to generate.

    Returns:
        list: Simulated demands for each product.
    """
    demands = []
    for price in prices:
        alpha, beta = generate_price_demand_params()
        noise = np.random.normal(0, 10, n_samples)
        demand = alpha - beta * price + noise
        demands.append(np.maximum(demand, 0))  # Ensure non-negative demand
    return demands
