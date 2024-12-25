import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_cutoff_for_stability(revenues, epsilon=0.005, rolling_window=50, results_path="results"):
    """
    Identify the first N scenarios to cut based on cumulative and rolling average stability.

    Parameters:
        revenues (list or np.ndarray): Array of revenue values for all scenarios.
        epsilon (float): Threshold for stability as a fraction of the mean (default: 0.01).
        rolling_window (int): Window size for rolling averages (default: 50).
        results_path (str): Path to save the stability analysis plot (default: "results").

    Returns:
        int: The number of initial scenarios to cut.
    """
    revenues = np.array(revenues)

    # Compute cumulative averages
    cumulative_avg = np.cumsum(revenues) / np.arange(1, len(revenues) + 1)

    # Compute rolling averages
    rolling_avg = pd.Series(revenues).rolling(rolling_window).mean()

    # Stability condition: cumulative average change < epsilon * mean
    stable_cumulative = np.abs(np.diff(cumulative_avg)) < epsilon * cumulative_avg[-1]
    stable_cumulative_index = np.argmax(stable_cumulative) + 1 if np.any(stable_cumulative) else len(revenues)

    # Stability condition: rolling average change < epsilon * mean
    stable_rolling = rolling_avg.dropna().diff().abs() < epsilon * cumulative_avg[-1]
    stable_rolling_index = (
        np.argmax(stable_rolling.values) + rolling_window if np.any(stable_rolling) else len(revenues)
    )

    # Choose the maximum index for stability
    cut_n = max(stable_cumulative_index, stable_rolling_index)

    # Plot stability analysis
    plt.figure(figsize=(12, 6))

    # Plot revenues
    plt.plot(revenues, label="Revenue", alpha=0.5)

    # Plot cumulative average
    plt.plot(cumulative_avg, label="Cumulative Average", color="blue", linewidth=2)

    # Plot rolling average
    plt.plot(rolling_avg, label=f"Rolling Average (Window={rolling_window})", color="orange", linewidth=2)

    # Mark the cutoff point
    plt.axvline(cut_n, color="red", linestyle="--", label=f"Cut-off at Scenario {cut_n}")

    plt.title("Stability Analysis: Revenue Averages")
    plt.xlabel("Scenario")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}/stability_analysis.png")
    plt.close()

    return cut_n
