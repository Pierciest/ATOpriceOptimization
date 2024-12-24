import numpy as np
from config.parameters import NUM_MACHINES, TIME_AVAILABILITY_RANGE

def generate_machines(num_machines=NUM_MACHINES, time_availability_range=TIME_AVAILABILITY_RANGE):
    """
    Generate a random set of machines with their time availability.
    Args:
        num_machines (int): Number of machines to generate.
        time_availability_range (tuple): Min and max range for time availability in hours.

    Returns:
        dict: A dictionary with machine IDs as keys and time availability as values.
    """
    machine_availability = {
        f"Machine_{i+1}": np.random.uniform(*time_availability_range)
        for i in range(num_machines)
    }
    return machine_availability