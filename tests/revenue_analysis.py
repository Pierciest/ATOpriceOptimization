import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_revenue_files(folder="results"):
    """
    Load all revenue-related CSV files from the specified folder.
    """
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dataframes = {}
    for file in csv_files:
        model_name = file.replace("_demands.csv", "").replace("_", " ").title()
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        if "Revenue" in df.columns:
            dataframes[model_name] = df
            print(f"Loaded {model_name} revenue data from {file_path}")
        else:
            print(f"Skipping {file} as it does not contain revenue data.")
    return dataframes


def compute_revenue_statistics(df, model_name):
    """
    Compute key statistics (mean, variance) for revenues in a DataFrame.
    """
    revenue_stats = {
        "Mean Revenue": df["Revenue"].mean(),
        "Revenue Variance": df["Revenue"].var(),
    }
    print(f"\nRevenue Statistics for {model_name}:")
    for key, value in revenue_stats.items():
        print(f"{key}: {value:.2f}")
    return revenue_stats


def plot_revenue_distribution(df, model_name):
    """
    Plot the distribution of revenues for a given model.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Revenue"], kde=True, bins=30)
    plt.title(f"Revenue Distribution - {model_name}")
    plt.xlabel("Revenue")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def plot_revenue_comparison(revenue_stats):
    """
    Plot a comparison of average revenue and variance across models.
    """
    stats_df = pd.DataFrame(revenue_stats).T.reset_index()
    stats_df.columns = ["Model", "Mean Revenue", "Revenue Variance"]

    # Plot mean revenue
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Mean Revenue", data=stats_df, palette="Blues_d")
    plt.title("Mean Revenue by Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Revenue")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # Plot revenue variance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Revenue Variance", data=stats_df, palette="Oranges_d")
    plt.title("Revenue Variance by Model")
    plt.xlabel("Model")
    plt.ylabel("Revenue Variance")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def analyze_revenues(folder="results"):
    """
    Main function to analyze revenues across all demand models.
    """
    # Step 1: Load all revenue files
    dataframes = load_revenue_files(folder)

    # Step 2: Compute statistics for each model
    revenue_stats = {}
    for model_name, df in dataframes.items():
        stats = compute_revenue_statistics(df, model_name)
        revenue_stats[model_name] = stats

        # Step 3: Plot revenue distribution
        plot_revenue_distribution(df, model_name)

    # Step 4: Compare models
    plot_revenue_comparison(revenue_stats)


if __name__ == "__main__":
    analyze_revenues(folder="results")
