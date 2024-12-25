import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_files(folder="results"):
    """
    Load all CSV files from the results folder.
    """
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        model_name = file.replace("_demands.csv", "").replace("_", " ").title()
        file_path = os.path.join(folder, file)
        dataframes[model_name] = pd.read_csv(file_path)
        print(f"Loaded {model_name} data from {file_path}")
    return dataframes

def demand_summary(df, model_name):
    """
    Summarize demand statistics for a given model.
    """
    summary = df.groupby("Product")["Demand"].describe()
    print(f"\nSummary for {model_name}:\n")
    print(summary)
    return summary

def plot_demand_distribution(df, model_name):
    """
    Plot the distribution of demand for each product.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Demand", hue="Product", kde=True, multiple="stack", bins=30)
    plt.title(f"Demand Distribution - {model_name}")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")
    plt.legend(title="Product")
    plt.grid()
    plt.show()

def plot_demand_vs_price(df, model_name):
    """
    Plot demand vs. price for each product.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Price", y="Demand", hue="Product", alpha=0.6)
    plt.title(f"Demand vs Price - {model_name}")
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.legend(title="Product")
    plt.grid()
    plt.show()

def plot_total_demand(df, model_name):
    """
    Plot total demand across all products for each scenario.
    """
    total_demand = df.groupby("Scenario")["Demand"].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=total_demand, x="Scenario", y="Demand", marker="o")
    plt.title(f"Total Demand Across Scenarios - {model_name}")
    plt.xlabel("Scenario")
    plt.ylabel("Total Demand")
    plt.grid()
    plt.show()

def plot_product_stability(df, model_name):
    """
    Plot stability of demand across scenarios for each product.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Scenario", y="Demand", hue="Product", alpha=0.7)
    plt.title(f"Demand Stability Across Scenarios by Product - {model_name}")
    plt.xlabel("Scenario")
    plt.ylabel("Demand")
    plt.legend(title="Product")
    plt.grid()
    plt.show()

def compare_models(dataframes, product_id):
    """
    Compare demand distributions across models for a specific product.
    """
    comparison_data = []
    for model_name, df in dataframes.items():
        product_data = df[df["Product"] == product_id]
        product_data["Model"] = model_name
        comparison_data.append(product_data)

    comparison_df = pd.concat(comparison_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=comparison_df, x="Model", y="Demand")
    plt.title(f"Demand Distribution Comparison for Product {product_id}")
    plt.xlabel("Model")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def analyze_stability(df, num_scenarios):
    """
    Analyze stability of demand across scenarios.
    """
    stability = df.groupby("Scenario")["Demand"].std().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stability, x="Scenario", y="Demand")
    plt.title("Demand Stability Across Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Demand Standard Deviation")
    plt.grid()
    plt.show()

def analyze_csv_files(folder="results"):
    """
    Analyze all CSV files in the results folder.
    """
    dataframes = load_csv_files(folder)

    for model_name, df in dataframes.items():
        print(f"\nAnalyzing {model_name}...")
        # Summary statistics
        summary = demand_summary(df, model_name)

        # Plot distributions
        plot_demand_distribution(df, model_name)

        # Plot demand vs price
        plot_demand_vs_price(df, model_name)

        # Total demand across scenarios
        plot_total_demand(df, model_name)

        # Stability analysis by product
        plot_product_stability(df, model_name)

    # Compare models for a specific product (e.g., Product 1)
    product_id = 1
    print(f"\nComparing models for Product {product_id}...")
    compare_models(dataframes, product_id)

if __name__ == "__main__":
    analyze_csv_files(folder="results")
