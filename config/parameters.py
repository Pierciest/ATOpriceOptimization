# config/parameters.py

# Global Parameters
RANDOM_SEED = 42
NUM_COMPONENTS = 50
NUM_MACHINES = 3
NUM_PRODUCTS = 2
NUM_SCENARIOS = 500

# Ranges
COST_RANGE = (1, 1000)
TIME_AVAILABILITY_RANGE = (5, 20)
PRODUCTION_TIME_RANGE = (0.5, 5)
GOZINTO_FACTOR_RANGE = (1, 5)

# Set the random seed globally
import numpy as np
np.random.seed(RANDOM_SEED)
