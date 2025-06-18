# Stochastic Optimization for Assemble-to-Order (ATO) Problem

## Overview

This project presents a metamodel-based framework for **profit-maximizing price optimization** in Assemble-to-Order (ATO) systems under demand uncertainty. By integrating scenario generation, explicit inventory feasibility, and response-surface surrogates, we balance realism (component costs and stock constraints) with computational tractability.

## Features

1. **Scenario Generation:**

   * Create demand scenarios via moment matching to match target moments (mean, variance, skewness, kurtosis).
   * Refine scenarios by minimizing empirical Wasserstein distance to a target distribution.

2. **Inventory Management:**

   * InventoryManager tracks component stock levels and enforces feasibility constraints during simulation.
   * Calculates per-unit production costs based on component usage and flags infeasible assemblies.

3. **Demand Simulation:**

   * Linear price–demand model with additive noise and scenario-specific effects.
   * Simulates stochastic demand under each scenario, subject to inventory availability.

4. **Profit Estimation & Scenario Stability:**

   * Compute sample-average profit: (∑\_j (P\_j - c\_j) d\_j), subtracting per-unit component costs.
   * Adaptive sample-size selection and tolerance-based checks ensure stability of profit estimates across scenarios.

5. **Surrogate Modeling:**

   * Fit polynomial regression (degree min(3, J)) and neural-network (MLPRegressor) metamodels to the simulated profit surface.

6. **Optimization:**

   * Maximize surrogate-predicted profit over feasible price bounds using scipy.optimize.minimize (L-BFGS-B with SLSQP fallback).
   * Impose feasibility penalties for inventory violations or prices below production cost.

7. **Visualization:**

   * 2D heatmaps, contour plots, and 3D surface/scatter visualizations for 2- and 3-product cases.
   * Scalability plots: runtime vs. number of products.
   * Profit comparison plots: enumerated/sampled vs. optimized (polynomial and neural net).

## File Structure

```
.
├── ato.py   # Main class with all methods
├── utils.py                     # Helper functions for sampling and visualization
├── results/                     # Stores runtime, revenue comparisons, and visualizations
├── 2products.ipynb              # Playground for 2 products analysis
├── 3products.ipynb              # playground for 3 product analysis
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Python dependencies
└── stochastic_opt.pdf           # Report
```

## Key Concepts

* **Assemble-to-Order (ATO):** Components are stocked in advance; final products assemble upon order, allowing flexibility but requiring inventory management.
* **Stochastic Optimization:** Decision variables (prices) optimized against random demand scenarios to maximize expected profit.
* **Response-Surface Methods:** Fit inexpensive surrogate models to expensive profit evaluations and optimize these surrogates efficiently.

## Methodology

1. **Generate Scenarios:** Moment matching → Wasserstein-distance refinement → adaptive sample-size selection for stability.
2. **Simulate Profit:** For each price vector, simulate demand under scenarios, enforce inventory, and compute sample-average profit.
3. **Fit Metamodels:** Polynomial regression and neural-network surrogates approximate the profit surface.
4. **Optimize Surrogates:** Find price vectors maximizing surrogate predictions, applying penalties for infeasibility.
5. **Validate:** Re-evaluate optimized prices on fresh demand scenarios to estimate true expected profit gains.
6. **Scale:** Repeat for product counts $J=2$ through $J=10$, using full enumeration for $J\le4$ and random sampling for higher dimensions.

## Usage

1. **Prerequisites:** Python 3.8+ and required packages:

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the optimization:**

   ```bash
   python ato.py
   ```
3. **Outputs:**

   * Profit and runtime summaries printed to the console for each $J$.
   * Interactive plots display heatmaps, surfaces, and runtime/profit comparison charts.

## Results

* **Profit Improvements:** Polynomial surrogates consistently yield 1–2% profit gains over raw enumeration or sampling; neural networks provide comparable performance in low-dimensional cases.
* **Scalability:** Full enumeration is tractable up to $J=4$; random sampling scales the framework up to $J=10$ with runtimes under 6 minutes.

## License

This work is licensed for educational and research purposes only. Redistribution or commercial use is prohibited.
