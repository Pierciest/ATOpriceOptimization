import numpy as np
from src.simulation import independent_price_demand

def preliminary_simulation(price_min, price_max, manufacturing_costs, num_price_samples, scenario_steps):
    """
    Run a preliminary simulation to estimate revenue distributions.

    Parameters:
        price_min (list): Minimum prices for products.
        price_max (list): Maximum prices for products.
        manufacturing_costs (list): Manufacturing costs for products.
        num_price_samples (int): Number of price sets to generate.
        scenario_steps (int): Number of scenarios to simulate for each price set.

    Returns:
        list: Preliminary revenue samples.
    """
    preliminary_revenues = []
    price_sets = np.random.uniform(price_min, price_max, size=(num_price_samples, len(price_min)))

    for price_set in price_sets:
        for _ in range(scenario_steps):
            demands = independent_price_demand(price_set)
            revenue = np.sum(demands * price_set) - np.sum(manufacturing_costs)
            preliminary_revenues.append(revenue)

    return preliminary_revenues

def estimate_revenue_stability(revenues, confidence=0.95):
    """
    Estimate mean revenue and confidence intervals.

    Parameters:
        revenues (list): List of simulated revenues.
        confidence (float): Confidence level for intervals (default: 0.95).

    Returns:
        tuple: Mean, lower bound, upper bound of the confidence interval.
    """
    revenues = np.array(revenues)
    mean = np.mean(revenues)
    std_err = np.std(revenues, ddof=1) / np.sqrt(len(revenues))
    z = 1.96  # Z-value for 95% confidence
    margin = z * std_err
    return mean, mean - margin, mean + margin

def calculate_required_scenarios(std_dev, mean_revenue, margin_of_error=0.05, confidence=0.95):
    """
    Calculate the required number of scenarios for a given margin of error.

    Parameters:
        std_dev (float): Standard deviation of revenue.
        mean_revenue (float): Mean revenue.
        margin_of_error (float): Desired relative margin of error (default: 0.05).
        confidence (float): Confidence level for intervals (default: 0.95).

    Returns:
        int: Required number of scenarios.
    """
    z = 1.96  # Z-value for 95% confidence
    required_n = ((z * std_dev) / (margin_of_error * mean_revenue))**2
    return int(np.ceil(required_n))
