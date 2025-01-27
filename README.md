# Stochastic Optimization for Assemble-to-Order (ATO) Problem

## Overview
This project explores the use of stochastic optimization techniques to solve an Assemble-to-Order (ATO) problem. The key focus is on leveraging simulation and response surface methods to determine optimal pricing strategies while considering demand uncertainty and other stochastic factors.

---

## Features
1. **Scenario Generation:**
   - Uses moment matching and Wasserstein distance minimization techniques to generate scenarios that accurately represent demand distributions.

2. **Demand Simulation:**
   - Implements linear demand models affected by price sensitivity, noise, and scenario-specific effects.

3. **Revenue Estimation:**
   - Calculates revenue based on simulated demand data and given price combinations.

4. **Surrogate Models:**
   - Fits both polynomial regression and neural network models to approximate the revenue surface.

5. **Optimization:**
   - Optimizes prices using surrogate models and evaluates them using scenario-based demand simulations.

6. **Visualization:**
   - 2D and 3D plots for analyzing revenue surfaces, contours, and scatter data for up to 3 products.
   - Scalability analysis of runtime and revenue comparison.

---

## File Structure

```
.
├── stochastic_optimization.py   # Main class with all methods
├── utils.py                     # Helper functions for sampling and visualization
├── results/                     # Stores runtime, revenue comparisons, and visualizations
├── 2products.ipynb              # Playground for 2 products analysis
├── 3products.ipynb              # playground for 3 product analysis
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Python dependencies
```

---

## Key Concepts

### Assemble-to-Order (ATO) Problem
In an ATO system, components are produced in advance and assembled into final products only after customer orders are received. This allows greater flexibility in responding to uncertain demand.

### Stochastic Optimization
Optimization under uncertainty is performed by:
1. Generating scenarios to model possible outcomes.
2. Optimizing decision variables to maximize expected revenue while accounting for the stochastic nature of demand.

---

## Methodology

### 1. Scenario Generation
Scenarios are created using:
- **Moment Matching:** Ensures the scenarios match desired statistical properties (mean, variance, skewness, and kurtosis).
- **Wasserstein Distance Minimization:** Refines scenarios to closely resemble a target distribution.

### 2. Demand Simulation
Simulates stochastic demand for products based on:
- **Price Sensitivity:** Linear model linking demand to price.
- **Scenario Effects:** Incorporates scenario-specific means and variances.
- **Noise:** Adds randomness to mimic real-world variations.

### 3. Revenue Surface Approximation
Fits surrogate models to the simulated revenue data:
- **Polynomial Regression:** Provides interpretable approximations of revenue surfaces.
- **Neural Networks:** Captures complex, nonlinear relationships for more accurate predictions.

### 4. Price Optimization
Optimizes prices to maximize revenue by solving:
- `max p R(p)` where `R(p)` is the revenue predicted by the surrogate model.

### 5. Scalability and Analysis
- Measures runtime as the number of products increases.
- Compares revenue improvements from polynomial and neural network optimizations.

---

## Visualization
- **2D Heatmaps and Contours:** For 2-product scenarios.
- **3D Surface Plots:** For 2- and 3-product scenarios.
- **Scalability Plots:** Runtime vs. number of products.
- **Revenue Comparison Plots:** Baseline vs. optimized revenues (Polynomial and Neural Network models).

---

## Usage

### Prerequisites
1. Install Python 3.8+.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Main Script
1. Open `stochastic_optimization.py`.
2. Modify the `product_list` variable to set the number of products to analyze.
3. Run the script:
   ```bash
   python stochastic_optimization.py
   ```

### Output
1. Simulation data and visualizations are saved in the `results/` folder.
2. Runtime and revenue metrics are printed for each product count.

---

## Results
### Scalability
The runtime scales polynomially with the number of products due to increased dimensionality in price combinations.

### Revenue Optimization
- **Polynomial Models:** Efficient for moderately complex problems.
- **Neural Networks:** Offer superior performance for high-dimensional problems.

---

## Acknowledgments
This project is based on lecture notes and coursework by Edoardo Fadda (Politecnico di Torino) on stochastic optimization and simulation methods.

---

## License
This project is licensed for educational and research purposes only. Redistribution or commercial use is prohibited.
