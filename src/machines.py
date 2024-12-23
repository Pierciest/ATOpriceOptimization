import numpy as np
from config.parameters import NUM_MACHINES, TIME_AVAILABILITY_RANGE

def generate_machines():
    """
    Generate machines with their time availability.
    Returns:
        dict: Machine IDs mapped to time availability.
    """
    return {f"Machine_{i+1}": np.random.uniform(*TIME_AVAILABILITY_RANGE) for i in range(NUM_MACHINES)}
