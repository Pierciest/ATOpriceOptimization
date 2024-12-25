import os
import pandas as pd

def save_data_to_examples(components, gozinto_matrix, demand_data, scenario_revenues, avg_revenues, examples_path="examples"):
    """
    Saves various simulation data to the specified examples folder.
    """
    os.makedirs(examples_path, exist_ok=True)

    # Save product costs
    components_df = pd.DataFrame(list(components.items()), columns=['Component', 'Cost'])
    components_df.to_csv(f"{examples_path}/product_costs.csv", index=False)

    # Save Gozinto matrix
    gozinto_df = pd.DataFrame(gozinto_matrix)
    gozinto_df.to_csv(f"{examples_path}/gozinto_matrix.csv", index=False, header=False)

    # Save demand data for each price set and scenario
    demand_data.to_csv(f"{examples_path}/demand_data.csv", index=False)

    # Save revenue for each scenario
    scenario_revenues.to_csv(f"{examples_path}/scenario_revenues.csv", index=False)

    # Save average revenues after cut
    avg_revenues.to_csv(f"{examples_path}/average_revenues.csv", index=False)

    print("Data saved to examples folder.")
