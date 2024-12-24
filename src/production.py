import numpy as np
from config.parameters import (
    NUM_COMPONENTS,
    NUM_PRODUCTS,
    NUM_MACHINES,
    PRODUCTION_TIME_RANGE,
    GOZINTO_FACTOR_RANGE,
)

def generate_production_times(num_components = NUM_COMPONENTS, num_machines = NUM_MACHINES, time_range= PRODUCTION_TIME_RANGE):
    """
    Generate the time required to produce each component on each machine.
    Args:
        num_components (int): Number of components.
        num_machines (int): Number of machines.
        time_range (tuple): Min and max range for production times in hours.

    Returns:
        np.ndarray: A matrix of shape (num_components, num_machines) with production times.
    """
    production_times = np.random.uniform(*time_range, size=(num_components, num_machines))
    return production_times


def generate_gozinto_factors(num_components = NUM_COMPONENTS, num_products = NUM_PRODUCTS, factor_range=(1, 5)):
    """
    Generate the Gozinto factor matrix linking components to products.
    Args:
        num_components (int): Number of components.
        num_products (int): Number of products.
        factor_range (tuple): Min and max range for Gozinto factors.

    Returns:
        np.ndarray: A matrix of shape (num_components, num_products) with Gozinto factors.
    """
    gozinto_matrix = np.random.randint(*factor_range, size=(num_components,num_products))
    return gozinto_matrix
