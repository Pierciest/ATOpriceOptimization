from src.components import generate_components
from src.production import generate_gozinto_factors
from src.demand import simulate_demand
from src.revenue_analysis import evaluate_revenue
from src.visualization import plot_revenue_vs_scenarios
import numpy as np

def main():
    # Generate components and Gozinto matrix
    components = generate_components()
    gozinto_matrix = generate_gozinto_factors()
    component_costs = np.array(list(components.values()))

    # Generate price sets
    price_sets = [[np.random.uniform(cost + 5, cost + 50) for cost in component_costs] for _ in range(5)]

    # Simulate scenarios and evaluate revenues
    max_scenarios = 500
    scenario_steps = np.arange(10, max_scenarios + 1, 10)
    revenue_results = {tuple(prices): [] for prices in price_sets}

    for num_scenarios in scenario_steps:
        demand_scenarios = np.random.randint(50, 150, size=(num_scenarios, len(price_sets)))
        for price_set in price_sets:
            avg_revenue = evaluate_revenue(price_set, demand_scenarios, gozinto_matrix, component_costs)
            revenue_results[tuple(price_set)].append(avg_revenue)

    # Visualize results
    plot_revenue_vs_scenarios(scenario_steps, revenue_results)

if __name__ == "__main__":
    main()
