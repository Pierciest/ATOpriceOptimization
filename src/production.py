import numpy as np
from config.parameters import (
    NUM_COMPONENTS,
    NUM_PRODUCTS,
    PRODUCTION_TIME_RANGE,
    GOZINTO_FACTOR_RANGE,
)

def generate_production_times():
    """
    Generate production times for components on machines.
    Returns:
        np.ndarray: Matrix of production times.
    """
    return np.random.uniform(*PRODUCTION_TIME_RANGE, size=(NUM_COMPONENTS, NUM_MACHINES))

def generate_gozinto_factors():
    """
    Generate the Gozinto matrix linking components to products.
    Returns:
        np.ndarray: Gozinto factor matrix.
    """
    return np.random.randint(*GOZINTO_FACTOR_RANGE, size=(NUM_PRODUCTS, NUM_COMPONENTS))
