import numpy as np
from config.parameters import NUM_COMPONENTS, COST_RANGE

def generate_components():
    """
    Generate components with their associated costs.
    Returns:
        dict: Component IDs mapped to costs.
    """
    return {f"Component_{i+1}": np.random.uniform(*COST_RANGE) for i in range(NUM_COMPONENTS)}
