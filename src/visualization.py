import matplotlib.pyplot as plt

def save_demand_and_revenue_distributions(demand_df, avg_revenues_df, results_path):
    # Plot distribution of demands for the product depending on prices
    plt.figure(figsize=(10, 6))
    for product in range(len(demand_df['Demands'].iloc[0])):  # Use length of the first list
        demands = demand_df['Demands'].apply(lambda x: x[product])
        # Extract prices if available, otherwise use a placeholder or calculation
        prices = demand_df['PriceSet'] if 'PriceSet' in demand_df else range(len(demands))
        plt.scatter(prices, demands, label=f'Product {product + 1} Demands', alpha=0.6)
    plt.title("Distribution of Demands Depending on Prices")
    plt.xlabel("Price or PriceSet")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/demand_distribution.png")
    plt.close()

    # Plot revenue distribution depending on prices
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_revenues_df['P1'], avg_revenues_df['Revenue'], label='Revenue vs P1', color='blue', alpha=0.6)
    plt.scatter(avg_revenues_df['P2'], avg_revenues_df['Revenue'], label='Revenue vs P2', color='red', alpha=0.6)
    plt.title("Revenue Distribution Depending on Prices")
    plt.xlabel("Prices")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/revenue_distribution.png")
    plt.close()



def save_revenue_plots(revenue_df, results_path, scenario_revenues=None):
    # Dot plot for average revenues
    plt.figure(figsize=(10, 6))
    plt.scatter(revenue_df['P1'], revenue_df['Revenue'], label='P1 Revenue', color='blue', alpha=0.6, edgecolors='w', s=80)
    plt.scatter(revenue_df['P2'], revenue_df['Revenue'], label='P2 Revenue', color='red', alpha=0.6, edgecolors='w', s=80)
    plt.title("Revenue vs. Prices")
    plt.xlabel("Prices")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/dot_plot_revenues.png")
    plt.close()

    # Scenario revenues plot (if provided)
    if scenario_revenues is not None:
        sampled_scenarios = scenario_revenues.groupby('PriceSet').sample(n=5, random_state=42)
        plt.figure(figsize=(10, 6))
        for price_set, group in sampled_scenarios.groupby('PriceSet'):
            plt.plot(group['Scenario'], group['Revenue'], label=f'PriceSet {price_set}', marker='o', linestyle='-', alpha=0.7)
        plt.title("Revenue vs. Scenarios (Sampled)")
        plt.xlabel("Scenario")
        plt.ylabel("Revenue")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_path}/scenario_vs_revenue_sampled.png")
        plt.close()