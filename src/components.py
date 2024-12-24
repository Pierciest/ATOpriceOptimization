import numpy as np
from config.parameters import NUM_COMPONENTS, COST_RANGE

def generate_components(num_components=NUM_COMPONENTS, cost_range=COST_RANGE):
    """
    Generate a random set of components with their associated costs.
    Args:
        num_components (int): Number of components to generate.
        cost_range (tuple): Min and max range for component costs.

    Returns:
        dict: A dictionary with component IDs as keys and costs as values.
    """
    component_costs = {
        f"Component_{i+1}": np.random.uniform(*cost_range)
        for i in range(num_components)
    }
    return component_costs