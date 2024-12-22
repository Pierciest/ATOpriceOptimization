# ATO Demand Analysis

This repository contains an implementation of an **Assemble-to-Order (ATO) Demand Analysis** model for optimizing product prices to maximize revenue under stochastic demand scenarios. The model incorporates stochastic simulation, response surface modeling, and optimization techniques to derive actionable insights for price-demand management.

---

## Project Structure

```
ATOdemandAnalysis/
|
├── instances/               # Contains instance-specific data
│   ├── __init__.py         # Package initialization
│   ├── coefficients.json   # Demand coefficients
│
├── settings/                # Global parameters
│   ├── parameters.py       # Defines constants like scenario size and price range
│
├── solvers/                 # Core modules for the project
│   ├── demand.py           # Stochastic demand generation
│   ├── ato_model.py        # ATO model logic and constraints
│   ├── response_surface.py # Response surface modeling and optimization
│   ├── visualize.py        # Visualization utilities
│
├── example.ipynb            # Interactive exploration notebook
├── tests/                   # Test modules
│   ├── test_demand.py      # Unit tests for demand generation
│   ├── test_model.py       # Unit tests for ATO model
│   ├── test_response.py    # Unit tests for response surface
│
├── main.py                  # Main script to execute price-demand optimization
├── main_stability.py        # Script to analyze in-sample and out-of-sample stability
├── README.md                # Project overview (this file)
└── requirements.txt         # Python dependencies
```

---

## Features

### 1. Stochastic Demand Simulation
- Simulates demand scenarios based on price sensitivity coefficients and randomness.
- Outputs scenarios including prices, demands, and corresponding revenues.

### 2. Assemble-to-Order (ATO) Model
- Implements ATO constraints such as component availability and demand fulfillment.
- Simplifies optimization to focus on price-revenue dynamics.

### 3. Response Surface Modeling
- Fits a regression model to approximate the relationship between prices and revenue.
- Enables fast optimization for price decisions.

### 4. Stability Analysis
- **In-Sample Stability**: Evaluates consistency of solutions across independent scenario sets.
- **Out-of-Sample Stability**: Tests optimized solutions on larger validation datasets.

### 5. Visualization
- Generates 3D surface plots and 2D contour maps for revenue as a function of prices.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pierciest/ATOdemandAnalysis.git
   cd ATOdemandAnalysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### **Price-Demand Optimization**
Run the main script to simulate scenarios, build the response surface, and optimize prices:
```bash
python main.py
```

### **Stability Analysis**
Analyze stability of results using `main_stability.py`:
```bash
python main_stability.py
```

### **Interactive Exploration**
Use the `example.ipynb` notebook to explore scenarios, visualize surfaces, and fine-tune parameters interactively.

---

## Parameters
Modify the `settings/parameters.py` file to adjust global parameters:
```python
NUM_PRODUCTS = 2       # Number of products
NUM_SCENARIOS = 100    # Number of demand scenarios
SEED = 42              # Random seed for reproducibility
PRICE_RANGE = (10, 50) # Range of product prices
```

---

## Results

### Example Output from `main.py`:
- **Optimal Prices**: `[35.4, 28.7]`
- **Revenue Surface**:
  - 3D plot showing how revenue changes with prices.
  - 2D contour map highlighting optimal price regions.

### Example Output from `main_stability.py`:
- **In-Sample Stability**:
  ```plaintext
  In-Sample Stability for size 50: Avg Difference = 3.25
  In-Sample Stability for size 100: Avg Difference = 1.57
  ```
- **Out-of-Sample Revenue**:
  ```plaintext
  Out-of-Sample Revenue (Validation): 3025.34
  ```

---

## Testing

Run the test suite to validate functionality:
```bash
pytest tests/
```

---

## Dependencies
- Python 3.8+
- Key libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `pytest` (for testing)

---

## Future Enhancements
- Extend the ATO model to include more complex constraints.
- Implement advanced metamodeling techniques like neural networks.
- Add interactive visualizations using Plotly.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For any questions or contributions, please reach out to:
- **Name**: Pierciest
- **GitHub**: [Pierciest](https://github.com/Pierciest)

