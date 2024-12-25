import os
import numpy as np
from src.simulation import run_simulation
from src.components import generate_components
from src.production import generate_gozinto_factors
from src.visualization import save_revenue_plots, save_demand_and_revenue_distributions
from models.metamodel import fit_metamodel, load_metamodel
from models.ato_model import save_ato_model, load_ato_model
from solvers.optimization import optimize_prices
from surface_response.response_analysis import fit_and_visualize_surface
from src.data_saver import save_data_to_examples
from tests.confidence_test import preliminary_simulation, estimate_revenue_stability, calculate_required_scenarios
from tests.stability_analysis import find_cutoff_for_stability

def main():
    np.random.seed(42)

    # Generate components and their costs
    print("Generating components and costs...")
    components = generate_components()
    component_costs = np.array(list(components.values()))

    # Generate Gozinto matrix and calculate manufacturing costs
    print("Generating Gozinto matrix and calculating manufacturing costs...")
    gozinto_matrix = generate_gozinto_factors()
    manufacturing_costs = gozinto_matrix.T @ component_costs

    # Generate price ranges based on manufacturing costs
    print("Generating price ranges...")
    low_prices = [np.random.uniform(cost, cost * 4) for cost in manufacturing_costs]
    high_prices = [np.random.uniform(cost * 4, cost * 8) for cost in manufacturing_costs]

    # Perform preliminary simulation to determine required scenarios
    print("Performing preliminary simulation to estimate stability...")
    preliminary_revenues = preliminary_simulation(
        price_min=low_prices,
        price_max=high_prices,
        manufacturing_costs=manufacturing_costs,
        num_price_samples=10,  # Use a smaller number of samples for preliminary simulation
        scenario_steps=50  # Use fewer steps for preliminary analysis
    )

    # Estimate revenue stability and required scenarios
    mean_revenue, lower_bound, upper_bound = estimate_revenue_stability(preliminary_revenues)
    std_dev = np.std(preliminary_revenues, ddof=1)
    required_scenarios = calculate_required_scenarios(std_dev, mean_revenue, margin_of_error=0.05)
    print(f"Preliminary Mean Revenue: {mean_revenue}")
    print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")
    print(f"Estimated Required Number of Scenarios: {required_scenarios}")

    # Determine the first N scenarios to cut based on stability
    print("Analyzing stability to determine scenarios to cut...")
    cut_n = find_cutoff_for_stability(preliminary_revenues)
    print(f"Determined Cut-off for Stability: First {cut_n} scenarios to cut.")

    # Load or create ATO model configuration
    config_path = "config/ato_model_config.json"
    os.makedirs("config", exist_ok=True)

    if os.path.exists(config_path):
        print(f"Loading ATO model configuration from {config_path}...")
        config = load_ato_model(config_path)
    else:
        print("No saved configuration found. Using default parameters.")
        config = {
            "price_min": low_prices,
            "price_max": high_prices,
            "manufacturing_costs": manufacturing_costs.tolist(),
            "num_price_samples": 50,
            "scenario_steps": int(required_scenarios),  # Convert to native int
            "cut_n": int(cut_n)  # Convert to native int
        }

        save_ato_model(config, config_path)
        print(f"Configuration saved to {config_path}.")

    # Run simulation
    print("Running simulation...")
    avg_revenues_df, demand_df, scenario_revenues_df, uncut_revenues_df = run_simulation(
        price_min=config["price_min"],
        price_max=config["price_max"],
        manufacturing_costs=config["manufacturing_costs"],
        num_price_samples=config["num_price_samples"],
        scenario_steps=config["scenario_steps"],
        cut_n=config["cut_n"]
    )
    print("Simulation complete. Sample revenue data:")
    print(avg_revenues_df.head())

    # Find and display the highest revenue in the simulation
    max_revenue_row = avg_revenues_df.loc[avg_revenues_df['Revenue'].idxmax()]
    print("Highest Revenue Found During Simulation:")
    print(f"PriceSet: {max_revenue_row['PriceSet']}, Revenue: {max_revenue_row['Revenue']}")

    # Load or fit metamodel
    metamodel_path = "models/metamodel.pkl"
    if os.path.exists(metamodel_path):
        print(f"Loading saved metamodel from {metamodel_path}...")
        model, scaler_X, scaler_y = load_metamodel(metamodel_path)
    else:
        print("Fitting and saving new metamodel...")
        model, scaler_X, scaler_y = fit_metamodel(avg_revenues_df, save_path=metamodel_path)
        print(f"Metamodel saved to {metamodel_path}.")

    # Perform surface response analysis
    print("Performing surface response analysis...")
    model, scaler_X, scaler_y, response_function = fit_and_visualize_surface(avg_revenues_df, results_path="results")
    print("Surface response analysis complete. Plots saved to results folder.")

    # Optimize prices using the metamodel
    print("Optimizing prices using the metamodel...")
    optimal_x1, optimal_x2, optimal_revenue = optimize_prices(model, scaler_X, scaler_y)
    print(f"Optimal P1: {optimal_x1}")
    print(f"Optimal P2: {optimal_x2}")
    print(f"Maximum Predicted Revenue: {optimal_revenue}")

    # Test optimized prices in the ATO model
    print("Testing optimized prices in the ATO model...")
    optimized_prices = [optimal_x1, optimal_x2]
    optimized_revenue_df, _, _, _ = run_simulation(
        price_min=optimized_prices,
        price_max=optimized_prices,
        manufacturing_costs=config["manufacturing_costs"],
        num_price_samples=1,
        scenario_steps=config["scenario_steps"],
        cut_n=config["cut_n"]
    )
    print("Testing complete. Revenue at optimized prices:")
    print(optimized_revenue_df.head())

    # Save data to examples
    print("Saving all data to examples folder...")
    save_data_to_examples(
        components=components,
        gozinto_matrix=gozinto_matrix,
        demand_data=demand_df,
        scenario_revenues=scenario_revenues_df,
        avg_revenues=avg_revenues_df,
        examples_path="examples"
    )

    # Save uncut scenario revenues
    uncut_revenues_df.to_csv("examples/uncut_scenario_revenues.csv", index=False)

    # Save plots
    print("Saving additional plots to results folder...")
    save_revenue_plots(avg_revenues_df, results_path="results", scenario_revenues=scenario_revenues_df)

    # Save demand and revenue distributions
    save_demand_and_revenue_distributions(demand_df, avg_revenues_df, results_path="results")

    print("All plots and data saved.")

if __name__ == "__main__":
    main()
