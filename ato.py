import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Component:
    name: str
    inventory: int
    cost: float
    lead_time: int = 1  # Days to replenish

@dataclass
class ProductRecipe:
    name: str
    components: Dict[str, int]  # component_name -> quantity needed
    production_cost: float = 0.0  # Will be calculated from components

class InventoryManager:
    def __init__(self, components: List[Component], product_recipes: Dict[str, ProductRecipe]):
        self.components = {c.name: c for c in components}
        self.product_recipes = product_recipes
        self._calculate_production_costs()
    
    def _calculate_production_costs(self):
        """Calculate production cost for each product based on component costs"""
        for recipe in self.product_recipes.values():
            total_cost = 0
            for comp_name, quantity in recipe.components.items():
                comp_cost = self.components[comp_name].cost
                total_cost += comp_cost * quantity
            recipe.production_cost = total_cost
    
    def can_produce(self, product_name: str, quantity: int) -> bool:
        """Check if we have enough components to produce quantity of product"""
        recipe = self.product_recipes[product_name]
        for comp_name, needed_qty in recipe.components.items():
            total_needed = needed_qty * quantity
            if total_needed > self.components[comp_name].inventory:
                return False
        return True
    
    def get_max_production(self, product_name: str) -> int:
        """Calculate maximum possible production quantity for a product"""
        recipe = self.product_recipes[product_name]
        max_units = float('inf')
        for comp_name, needed_qty in recipe.components.items():
            available = self.components[comp_name].inventory
            possible_units = available // needed_qty
            max_units = min(max_units, possible_units)
        return int(max_units)
    
    def get_production_cost(self, product_name: str) -> float:
        """Get per-unit production cost for a product"""
        return self.product_recipes[product_name].production_cost
    
    def optimal_production_quantity(self, product_name: str, demand_mean: float, 
                             demand_std: float) -> int:
        """
        Calculate optimal production quantity considering inventory constraints.
        
        Args:
            product_name: Name of the product
            demand_mean: Mean of predicted demand
            demand_std: Standard deviation of predicted demand
            
        Returns:
            int: Optimal production quantity constrained by component availability
        """
        max_units = self.get_max_production(product_name)
        optimal = min(max_units, int(demand_mean + demand_std * 1.96))  # 95% confidence
        return optimal
    
    def use_components(self, product_name: str, quantity: int) -> bool:
        """
        Attempt to use components for production. Returns True if successful.
        """
        if not self.can_produce(product_name, quantity):
            return False
            
        recipe = self.product_recipes[product_name]
        for comp_name, needed_qty in recipe.components.items():
            self.components[comp_name].inventory -= needed_qty * quantity
        return True
        

# -------------------------------------------------
# StochasticOptimization Class
# -------------------------------------------------
class StochasticOptimization:
    """
    Scenario generation, demand simulation, polynomial fitting, neural network fitting, and price optimization.
    """

    @staticmethod
    def moment_matching(mean, variance, skewness, kurtosis, num_scenarios):
        """
        Simple demonstration: perturb a Normal distribution by skewness/kurtosis.
        """
        z = np.random.normal(size=num_scenarios)
        scenarios = (
            mean
            + np.sqrt(variance)*z
            + skewness*(z**2 - 1)
            + kurtosis*(z**3 - 3*z)
        )
        return np.clip(scenarios, 0, None)

    @staticmethod
    def minimize_wasserstein_distance(target_distribution, initial_scenarios, max_iterations=100):
        """
        Simplistic random search to reduce Wasserstein distance.
        """
        current = initial_scenarios.copy()
        for _ in range(max_iterations):
            for i in range(len(current)):
                adjust = np.random.normal(scale=0.1)
                test = current.copy()
                test[i] += adjust
                if (wasserstein_distance(target_distribution, test)
                   < wasserstein_distance(target_distribution, current)):
                    current = test
        return current

    @staticmethod
    def generate_advanced_scenarios(mean, variance, skewness, kurtosis,
                                    num_scenarios, target_distribution,
                                    max_iterations=100):
        """
        1) moment_matching -> 2) minimize_wasserstein_distance
        """
        init = StochasticOptimization.moment_matching(
            mean, variance, skewness, kurtosis, num_scenarios
        )
        refined = StochasticOptimization.minimize_wasserstein_distance(
            target_distribution, init, max_iterations
        )
        return refined

    @staticmethod
    def generate_stochastic_demand_linear(prices, num_samples=1000, noise_std=5, scenarios=None, 
                                        inventory_manager=None):
        """Modified to consider inventory constraints and production costs"""
        J = len(prices)
        # More realistic base demands and price sensitivities
        base_demands = [1000 - 40*i for i in range(J)]  # Decreasing base demand for higher tiers
        price_sensitivities = []
        for i in range(J):
            if i < 3:  # Entry-level products (more elastic)
                sensitivity = 0.5 + 0.1*i
            else:  # Premium products (less elastic)
                sensitivity = 0.8 - 0.05*(i-3)
            price_sensitivities.append(max(0.3, min(0.9, sensitivity)))

        if inventory_manager:
            for i in range(J):
                product_name = f"Product_{i+1}"
                prod_cost = inventory_manager.get_production_cost(product_name)
                # Higher production cost -> lower price sensitivity
                price_sensitivities[i] *= 1.0 / (1 + prod_cost/2000)


        # If user doesn't provide scenarios, generate default
        if scenarios is None:
            target_dist = np.random.normal(loc=1.0, scale=0.5, size=1000)
            means_ = StochasticOptimization.generate_advanced_scenarios(
                mean=50, variance=100, skewness=0, kurtosis=3,
                num_scenarios=140, target_distribution=target_dist
            )
            scenarios = [
                {"mean": m, "variance": np.random.uniform(0.8,1.2)}
                for m in means_
            ]

        # Weighted scenario sampling
        w = np.ones(len(scenarios))
        p_ = w / w.sum()
        scenario_indices = np.random.choice(len(scenarios), size=num_samples, p=p_)

        demand_samples = np.zeros((num_samples, J))
        for i in range(J):
            alpha_i = base_demands[i]
            beta_i  = price_sensitivities[i]
            p_i     = prices[i]
            for row_idx, sc_idx in enumerate(scenario_indices):
                sc_mean = scenarios[sc_idx]['mean']
                sc_std  = np.sqrt(scenarios[sc_idx]['variance'])
                scenario_eff = np.random.normal(loc=sc_mean, scale=sc_std)
                noise = np.random.normal(loc=0, scale=noise_std)
                d_ij  = alpha_i - beta_i*p_i + scenario_eff + noise
                demand_samples[row_idx, i] = max(d_ij, 0)

        cols = [f"Product_{k+1}" for k in range(J)]
        return pd.DataFrame(demand_samples, columns=cols)

    @staticmethod
    def calculate_revenue(demand_data, prices):
        """
        Mean demand * price for each product, summed.
        """
        avg_dem = demand_data.mean(axis=0).values
        return np.dot(prices, avg_dem)
    
    @staticmethod
    def calculate_profit(demand_data, prices, inventory_manager=None):
        """New method: Calculate profit considering production costs"""
        revenue = StochasticOptimization.calculate_revenue(demand_data, prices)
        total_cost = 0
        
        if inventory_manager:
            avg_demand = demand_data.mean(axis=0)
            for i, demand in enumerate(avg_demand):
                product_name = f"Product_{i+1}"
                prod_cost = inventory_manager.get_production_cost(product_name)
                total_cost += prod_cost * demand
        
        return revenue - total_cost
    @staticmethod
    def fit_polynomial_surface(revenue_df, poly_degree=3):
        """
        revenue_df has columns [p1, p2, ..., pJ, profit].  # Updated docstring
        Returns (pipeline, y-scaler, response_function).
        """
        X = revenue_df.iloc[:, :-1].values  # price columns
        y = revenue_df.iloc[:, -1].values.reshape(-1, 1)

        scX = MinMaxScaler(feature_range=(-1,1))
        scY = MinMaxScaler(feature_range=(-1,1))

        pipe = Pipeline([
            ("scX", scX),
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("linreg", LinearRegression())
        ])

        y_sc = scY.fit_transform(y).ravel()
        pipe.fit(X, y_sc)

        def response(*price_arrays):
            """
            price_arrays: each is a 1D array of shape (n,) if evaluating multiple points.
            Returns predicted revenue(s) in original scale (unscaled).
            """
            inp = np.column_stack(price_arrays)
            X_ = pipe.named_steps["scX"].transform(inp)
            X_poly = pipe.named_steps["poly"].transform(X_)
            pred_sc = pipe.named_steps["linreg"].predict(X_poly)
            pred = scY.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            return pred

        return pipe, scY, response

    @staticmethod
    def fit_neural_net_surface(revenue_df, hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42):
        """
        Fit an MLPRegressor (neural net) to price/profit data.  # Updated docstring
        Returns (model, y-scaler, response_function).
        """
        X = revenue_df.iloc[:, :-1].values
        y = revenue_df.iloc[:, -1].values.reshape(-1, 1)

        scX = MinMaxScaler(feature_range=(-1,1))
        scY = MinMaxScaler(feature_range=(-1,1))

        X_scaled = scX.fit_transform(X)
        y_scaled = scY.fit_transform(y).ravel()

        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                           activation='relu',
                           solver='adam',
                           max_iter=max_iter,
                           random_state=random_state)
        mlp.fit(X_scaled, y_scaled)

        def response(*price_arrays):
            inp = np.column_stack(price_arrays)
            X_ = scX.transform(inp)
            pred_sc = mlp.predict(X_)
            pred = scY.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            return pred

        return mlp, scY, response


    @staticmethod
    def optimize_prices(response_function, bounds, method='L-BFGS-B', maxiter=500, inventory_manager=None):
        """
        Maximize profit by minimizing negative of the surrogate model's output.
        Added inventory manager to check constraints.
        """
        def objective(prices):
            try:
                val = response_function(*prices)
                
                # Apply inventory feasibility penalty
                if inventory_manager:
                    penalty = 0
                    for i, price in enumerate(prices):
                        product_name = f"Product_{i+1}"
                        max_prod = inventory_manager.get_max_production(product_name)
                        if max_prod <= 0:
                            penalty += 1e6  # Large penalty for infeasible solutions
                        
                        # Add penalty for prices below production cost
                        prod_cost = inventory_manager.get_production_cost(product_name)
                        if price < prod_cost:
                            penalty += 1e6 * (prod_cost - price)
                        
                    val -= penalty
                        
                return -val if isinstance(val, float) else -np.mean(val)
            except Exception:
                return 1e9  # Return large value if evaluation fails

        init_guess = [(b[0] + b[1])/2 for b in bounds]
        try:
            res = minimize(
                objective, init_guess, bounds=bounds, method=method,
                options={'maxiter': maxiter, 'maxfun': maxiter * 2}
            )
            if not res.success and method == 'L-BFGS-B':
                # Try again with different method if L-BFGS-B fails
                res = minimize(
                    objective, init_guess, bounds=bounds, method='SLSQP',
                    options={'maxiter': maxiter, 'maxfun': maxiter * 2}
                )
        except Exception:
            # Fallback to initial guess if optimization fails completely
            return init_guess, response_function(*init_guess)

        best_prices = res.x
        best_revenue_pred = response_function(*best_prices)
        if isinstance(best_revenue_pred, np.ndarray):
            best_revenue_pred = np.mean(best_revenue_pred)

        return best_prices, best_revenue_pred
# -------------------------------------------------
# Helper for random sampling of price combos
# -------------------------------------------------
def random_price_points(J, num_samples, bounds):
    """
    Generate 'num_samples' random combos in J-dim box.
    Now handles cases where bounds list might be incomplete.
    """
    if len(bounds) < J:
        print(f"[WARNING] Bounds list ({len(bounds)}) shorter than dimensions ({J})")
        return np.array([])  # Return empty array if bounds are insufficient
        
    points = []
    for _ in range(num_samples):
        row = []
        for j in range(J):
            lb, ub = bounds[j]
            if ub < lb:  # Check for invalid bounds
                print(f"[WARNING] Invalid bounds for dimension {j}: ({lb}, {ub})")
                return np.array([])
            val = np.random.uniform(lb, ub)
            row.append(val)
        points.append(row)
    return np.array(points)


def determine_scenario_count(base_scenarios, prices, inventory_manager, 
                           min_scenarios=50, max_scenarios=500, 
                           window_size=3, tolerance=0.01):
    """
    Determine appropriate number of scenarios by testing for profit stability.
    """
    profits = []
    counts = range(min_scenarios, max_scenarios + 50, 50)
    for n in counts:
        df = StochasticOptimization.generate_stochastic_demand_linear(
            prices=prices,
            num_samples=n,
            scenarios=base_scenarios[:n],
            inventory_manager=inventory_manager
        )
        profit = StochasticOptimization.calculate_profit(df, prices, inventory_manager)
        profits.append(profit)
        if len(profits) >= window_size:
            diffs = [abs(profits[i] - profits[i-1])/max(abs(profits[i-1]), 1e-10) 
                     for i in range(len(profits) - window_size + 1, len(profits))]
            if all(diff < tolerance for diff in diffs):
                return n
    return max_scenarios


# -------------------------------------------------
# Visualization for enumerated data (J=2 or J=3)
# -------------------------------------------------
def visualize_2d_heatmap_enumerated(profit_df):
    """For J=2 enumerated data, 2D contour/heatmap."""
    if profit_df.shape[1] != 3:
        return
    p1_unique = sorted(profit_df["p1"].unique())
    p2_unique = sorted(profit_df["p2"].unique())
    pivoted   = profit_df.pivot(index="p2", columns="p1", values="profit")
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)

    plt.figure(figsize=(8,6))
    cs = plt.contourf(p1_grid, p2_grid, pivoted.values, cmap='viridis', levels=20)
    plt.colorbar(cs)
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.title("Enumerated 2D Profit Heatmap (J=2)")
    plt.show()


def visualize_2d_surface_enumerated(profit_df):
    """For J=2 enumerated data, 3D surface plot."""
    if profit_df.shape[1] != 3:
        return
    p1_unique = sorted(profit_df["p1"].unique())
    p2_unique = sorted(profit_df["p2"].unique())
    pivoted   = profit_df.pivot(index="p2", columns="p1", values="profit")  # Changed from revenue
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)
    Z = pivoted.values

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(p1_grid, p2_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Profit")  # Changed from Revenue
    ax.set_title("Enumerated 3D Surface (J=2)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def visualize_2d_surface_poly(profit_df, response_func):
    """Evaluate response function and plot 3D surface + 2D contour."""
    if profit_df.shape[1] != 3:
        return
    X_ = profit_df.iloc[:, :-1]
    p1_vals = np.linspace(X_.iloc[:,0].min(), X_.iloc[:,0].max(), 50)
    p2_vals = np.linspace(X_.iloc[:,1].min(), X_.iloc[:,1].max(), 50)
    p1_grid, p2_grid = np.meshgrid(p1_vals, p2_vals)
    shape_ = p1_grid.shape

    flattened = np.stack([p1_grid.ravel(), p2_grid.ravel()], axis=-1)
    z_vals    = response_func(*flattened.T).reshape(shape_)

    # 3D surface
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(p1_grid, p2_grid, z_vals, cmap='viridis', edgecolor='none')
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Profit")  # Changed from Revenue
    ax.set_title("Model 3D Surface (J=2)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # 2D contour - restored
    plt.figure(figsize=(10,7))
    cs = plt.contourf(p1_grid, p2_grid, z_vals, cmap='viridis', levels=40)
    plt.colorbar(cs)
    plt.title("Model 2D Contour (J=2)")
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.show()



def visualize_3d_scatter_enumerated(profit_df):
    """For J=3 enumerated data, 3D scatter w/ color-coded profit."""
    if profit_df.shape[1] != 4:
        return
    X1 = profit_df["p1"].values
    X2 = profit_df["p2"].values
    X3 = profit_df["p3"].values
    Y  = profit_df["profit"].values  # Changed from revenue

    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(X1, X2, X3, c=Y, cmap='viridis', marker='o', alpha=0.8)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("p3")
    ax.set_title("Enumerated 3D Scatter (J=3)")
    cb = plt.colorbar(sc)
    cb.set_label("Profit")  # Changed from Revenue
    plt.show()


def visualize_3d_scatter_polynomial(profit_df, response_func, num_points=30):
    """
    Sample bounding box for J=3, show predicted profit as a 3D scatter.
    We can pass in the polynomial or NN response function.
    """
    if profit_df.shape[1] != 4:
        return

    p1_min, p1_max = profit_df["p1"].min(), profit_df["p1"].max()
    p2_min, p2_max = profit_df["p2"].min(), profit_df["p2"].max()
    p3_min, p3_max = profit_df["p3"].min(), profit_df["p3"].max()

    p1_rand = np.random.uniform(p1_min, p1_max, num_points)
    p2_rand = np.random.uniform(p2_min, p2_max, num_points)
    p3_rand = np.random.uniform(p3_min, p3_max, num_points)

    pred = response_func(p1_rand, p2_rand, p3_rand)

    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(p1_rand, p2_rand, p3_rand, c=pred, cmap='plasma', marker='o', alpha=0.8)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("p3")
    ax.set_title("Model 3D Scatter (J=3)")
    cb = plt.colorbar(sc)
    cb.set_label("Predicted Profit")  # Changed from Revenue
    plt.show()

def setup_inventory():
    """Create sample inventory configuration with realistic costs and constraints"""
    # Create basic components with realistic costs and inventory levels
    components = [
        Component("cpu", 1000, 200.0),      # High value component
        Component("memory", 2000, 100.0),    # Medium value component
        Component("storage", 1500, 50.0),    # Lower value component
        Component("case", 800, 30.0)         # Basic component
    ]
    
    # Create product recipes with increasing complexity/cost
    product_recipes = {}
    for i in range(1, 11):  # Products 1-10
        product_recipes[f"Product_{i}"] = ProductRecipe(
            name=f"Product_{i}",
            components={
                "cpu": 1,                    # Each product needs 1 CPU
                "memory": min(i, 8),         # Memory increases with tier, max 8 units
                "storage": min(i, 4),        # Storage increases with tier, max 4 units
                "case": 1                    # Each product needs 1 case
            }
        )
    
    return InventoryManager(components, product_recipes)

def calculate_bounds(prod_cost, i, base_demand=None, price_sensitivity=None):
    """
    Calculate bounds based on demand curve and profitability
    
    Args:
        prod_cost: Production cost
        i: Product index
        base_demand: Base demand for this product (alpha)
        price_sensitivity: Price sensitivity (beta)
    """
    # If demand parameters not provided, estimate them
    if base_demand is None:
        base_demand = 1000 - 40*i
    if price_sensitivity is None:
        if i < 3:
            price_sensitivity = 0.5 + 0.1*i
        else:
            price_sensitivity = 0.8 - 0.05*(i-3)
        price_sensitivity = max(0.3, min(0.9, price_sensitivity))
        price_sensitivity *= 1.0 / (1 + prod_cost/2000)

    # Calculate price where demand becomes zero
    p_zero_demand = base_demand / price_sensitivity
    
    # Calculate price where profit becomes zero (revenue = cost)
    # At price p: profit = p*demand - cost*demand = 0
    # demand = alpha - beta*p
    # p*(alpha - beta*p) = cost*(alpha - beta*p)
    # Solve for p: p = cost
    p_zero_profit = prod_cost
    
    # Lower bound: Ensure minimum markup
    min_markup = 1.2
    lb = max(prod_cost * min_markup, p_zero_profit)
    
    # Upper bound: Don't exceed zero demand point
    ub = min(p_zero_demand, p_zero_profit + (p_zero_demand - p_zero_profit)*0.9)
    
    # Add small padding to avoid numerical issues
    spread = ub - lb
    lb += spread * 0.05
    ub -= spread * 0.05
    
    return lb, ub

# -------------------------------------------------
# Main Script
# -------------------------------------------------
if __name__ == "__main__":
    np.random.seed(min(343381, 3434510, 339018, 348306))  # reproducibility

    inventory_manager = setup_inventory()

    start_time = time.time()  # Add this line

    # Updated product list
    product_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # We'll store results for plotting:
    runtimes = []
    best_enum_profits = []     # Changed from revenues
    best_opt_profits_poly = [] # Changed from revenues
    best_opt_profits_nn = []   # Changed from revenues

    # A helper to run the entire pipeline on a set of price combos
    def run_pipeline(price_grid, bounds, scenarios, J, label=""):
        """
        Given a grid of prices (N x J), compute scenario-based profit,
        fit polynomial + NN, optimize each, and compute real scenario-based profit
        at the optimized prices. Print results. Return dictionary of relevant stats.
        """
        # 1) Evaluate scenario-based profit for each price combo
        results_ = []

        initial_inventory = {name: comp.inventory \
                             for name, comp in inventory_manager.components.items()}

        for combo_ in price_grid:
            # Reset inventory before each evaluation
            for comp_name, initial_qty in initial_inventory.items():
                inventory_manager.components[comp_name].inventory = initial_qty

            df_ = StochasticOptimization.generate_stochastic_demand_linear(
                prices=combo_,
                num_samples=2000,
                noise_std=10,
                scenarios=scenarios,
                inventory_manager=inventory_manager
            )
            profit = StochasticOptimization.calculate_profit(df_, combo_, inventory_manager)
            results_.append(list(combo_) + [profit])  # Keep storing results

        col_names_ = [f"p{i+1}" for i in range(J)] + ["profit"]  # Change "revenue" to "profit"
        profit_df_ = pd.DataFrame(results_, columns=col_names_)

        # ----------------------------------------------------------------
        # (Optional) Visualization of enumerated data for J=2 or J=3
        # ----------------------------------------------------------------
        if J == 2:
            visualize_2d_heatmap_enumerated(profit_df_)
            visualize_2d_surface_enumerated(profit_df_)
        elif J == 3:
            visualize_3d_scatter_enumerated(profit_df_)

        # 2) Best enumerated/sampled
        idx_best_ = profit_df_["profit"].idxmax()  # Changed from "revenue" to "profit"
        best_prices_enum_ = profit_df_.loc[idx_best_, col_names_[:-1]].values
        best_profit_enum_ = profit_df_.loc[idx_best_, "profit"]  # Changed variable name

        print(f"({label}) Max profit (raw combos)    = {best_profit_enum_:.4f}")  # Updated message
        print(f"({label}) Prices that achieved it:     {best_prices_enum_}")

        # ---------------------------
        # 3) Polynomial Model
        # ---------------------------
        _, _, resp_fn_poly = StochasticOptimization.fit_polynomial_surface(
            profit_df_, poly_degree=min(2, J)  # Reduce polynomial degree for higher dimensions
        )
        opt_prices_poly, opt_profit_pred_poly = StochasticOptimization.optimize_prices(
            resp_fn_poly, bounds, method='L-BFGS-B', maxiter=500,
            inventory_manager=inventory_manager  # Add this parameter
        )
        # Evaluate real scenario-based profit at polynomial-optimized prices
        df_opt_poly = StochasticOptimization.generate_stochastic_demand_linear(
            prices=opt_prices_poly,
            num_samples=2000,
            noise_std=10,
            scenarios=scenarios,
            inventory_manager=inventory_manager
        )
        real_profit_opt_poly = StochasticOptimization.calculate_profit(
            df_opt_poly, opt_prices_poly, inventory_manager
        )

        print(f"({label}) [Poly] Predicted profit = {opt_profit_pred_poly:.4f}")
        print(f"({label}) [Poly] Real scenario-based profit = {real_profit_opt_poly:.4f}")

        if J == 2:
            visualize_2d_surface_poly(profit_df_, resp_fn_poly)
        elif J == 3:
            visualize_3d_scatter_polynomial(profit_df_, resp_fn_poly)

        # ---------------------------
        # 4) Neural Network Model
        # ---------------------------
        _, _, resp_fn_nn = StochasticOptimization.fit_neural_net_surface(
            profit_df_, hidden_layer_sizes=(128, 64, 32), max_iter=5000, random_state=42  # Deeper network, more iterations
        )
        opt_prices_nn, opt_profit_pred_nn = StochasticOptimization.optimize_prices(
            resp_fn_nn, bounds, method='L-BFGS-B', maxiter=500,
            inventory_manager=inventory_manager  # Add this parameter
        )
        # Evaluate real scenario-based profit at NN-optimized prices
        df_opt_nn = StochasticOptimization.generate_stochastic_demand_linear(
            prices=opt_prices_nn,
            num_samples=2000,
            noise_std=10,
            scenarios=scenarios,
            inventory_manager=inventory_manager
        )
        real_profit_opt_nn = StochasticOptimization.calculate_profit(
            df_opt_nn, opt_prices_nn, inventory_manager
        )

        print(f"({label}) [NN] Predicted profit = {opt_profit_pred_nn:.4f}")
        print(f"({label}) [NN] Real scenario-based profit = {real_profit_opt_nn:.4f}")

        if J == 2:
            visualize_2d_surface_poly(profit_df_, resp_fn_nn)
        elif J == 3:
            visualize_3d_scatter_polynomial(profit_df_, resp_fn_nn)

        # 5) Print baseline vs. optimized prices for reference
        baseline_prices_ = [0.5*(bnd[0] + bnd[1]) for bnd in bounds]
        print(f"({label}) Baseline prices:  {baseline_prices_}")
        print(f"({label}) [Poly] Optimized prices: {opt_prices_poly}")
        print(f"({label}) [NN]   Optimized prices: {opt_prices_nn}")
        for i_ in range(J):
            print(f"   Product {i_+1}: baseline={baseline_prices_[i_]:.2f}, "
                  f"poly={opt_prices_poly[i_]:.2f}, nn={opt_prices_nn[i_]:.2f}")

        # Return everything in a dictionary
        return {
            "best_profit_enum": best_profit_enum_,
            "best_prices_enum": best_prices_enum_,
            "opt_prices_poly": opt_prices_poly,
            "real_profit_opt_poly": real_profit_opt_poly,
            "opt_prices_nn": opt_prices_nn,
            "real_profit_opt_nn": real_profit_opt_nn,
            "profit_df": profit_df_
        }

    # -------------------------------------------------
    # Loop over different product counts
    # -------------------------------------------------
    initial_inventory = {name: comp.inventory for name, comp in inventory_manager.components.items()}

    for J in product_list:
        # Reset inventory levels at start of each iteration
        for comp_name, initial_qty in initial_inventory.items():
            inventory_manager.components[comp_name].inventory = initial_qty

        # 1) Generate scenario set for these J products
        print(f"[MAIN] Generating scenario set for {J} products...")
        target_dist = np.random.normal(loc=1.0, scale=0.5, size=1000)
        base_scenarios = [
            {"mean": m, "variance": np.random.uniform(0.8,1.2)}
            for m in StochasticOptimization.generate_advanced_scenarios(
                mean=50, variance=100, skewness=0.0, kurtosis=3.0,
                num_scenarios=500, target_distribution=target_dist
            )
        ]

        # Get initial price guesses for stability test
        initial_prices = []
        for i in range(J):
            product_name = f"Product_{i+1}"
            prod_cost = inventory_manager.get_production_cost(product_name)
            lb, ub = calculate_bounds(prod_cost, i)
            initial_prices.append((lb + ub) / 2)

        # Determine required number of scenarios
        optimal_count = determine_scenario_count(
            base_scenarios=base_scenarios,
            prices=initial_prices,
            inventory_manager=inventory_manager,
            min_scenarios=50,
            max_scenarios=500,
            window_size=3,
            tolerance=0.01
        )

        print(f"[MAIN] Determined optimal scenario count: {optimal_count}")
        shared_scenarios = base_scenarios[:optimal_count]

        # 2) Calculate price bounds considering inventory and production costs
        bounds = []
        feasible_products = []
        print(f"\nAnalyzing feasibility for J={J} products:")

        for i in range(J):
            product_name = f"Product_{i+1}"
            prod_cost = inventory_manager.get_production_cost(product_name)
            max_prod = inventory_manager.get_max_production(product_name)

            if max_prod <= 0:
                print(f"  {product_name}: Cannot produce (insufficient components)")
                lb_, ub_ = 1e6, 1e6 + 1
            else:
                print(f"  {product_name}: Can produce up to {max_prod} units")
                feasible_products.append(i)
                lb_, ub_ = calculate_bounds(prod_cost, i)
                print(f"    Cost: {prod_cost:.2f}, Bounds: ({lb_:.2f}, {ub_:.2f})")

            bounds.append((lb_, ub_))

        # 3) Decide approach based on J and feasibility
        if len(feasible_products) == 0:
            print(f"\n[WARNING] No feasible products for J={J}, skipping...")
            results_dict = {
                "best_profit_enum": 0,
                "real_profit_opt_poly": 0,
                "real_profit_opt_nn": 0
            }

        elif J < 4:
            step = 5
            ranges = []
            for (lb_, ub_) in bounds:
                if ub_ >= 1e6:
                    arr_ = np.array([ub_])
                else:
                    arr_ = np.linspace(lb_, ub_, step)
                ranges.append(arr_)

            grid = np.array(np.meshgrid(*ranges)).T.reshape(-1, J)
            feasible_mask = np.ones(len(grid), dtype=bool)
            for idx, prices in enumerate(grid):
                for j, price in enumerate(prices):
                    pname = f"Product_{j+1}"
                    if not inventory_manager.can_produce(pname, 1):
                        feasible_mask[idx] = False
                        break

            grid = grid[feasible_mask]
            if len(grid) > 0:
                print(f"\n[MAIN] Enumerating {grid.shape[0]} feasible combos for {J}D grid.")
                results_dict = run_pipeline(grid, bounds, shared_scenarios, J, label="Enumeration")
            else:
                print(f"\n[WARNING] No feasible combinations found for J={J}")
                results_dict = {"best_profit_enum": 0, "real_profit_opt_poly": 0, "real_profit_opt_nn": 0}

        elif J == 4:
            print("\n[MAIN] J=4 => Doing enumerations first.")
            step = 5
            ranges = []
            for (lb_, ub_) in bounds:
                if ub_ >= 1e6:
                    arr_ = np.array([ub_])
                else:
                    arr_ = np.linspace(lb_, ub_, step)
                ranges.append(arr_)

            grid_enum = np.array(np.meshgrid(*ranges)).T.reshape(-1, J)
            feasible_mask = np.ones(len(grid_enum), dtype=bool)
            for idx, prices in enumerate(grid_enum):
                for j, price in enumerate(prices):
                    pname = f"Product_{j+1}"
                    if not inventory_manager.can_produce(pname, 1):
                        feasible_mask[idx] = False
                        break

            grid_enum = grid_enum[feasible_mask]
            if len(grid_enum) > 0:
                print(f"   Enumerating {grid_enum.shape[0]} feasible combos.")
                _ = run_pipeline(grid_enum, bounds, shared_scenarios, J, label="Enum-Method")

            print("\n[MAIN] Now do random sampling approach for J=4.")
            num_samples_ = max(100, 42 * J)  # Increase samples for higher dimensions
            max_attempts = num_samples_ * 10

            feasible_points = []
            attempts = 0
            while len(feasible_points) < num_samples_ and attempts < max_attempts:
                point = random_price_points(J, 1, bounds)[0]
                is_feasible = True
                for j, price in enumerate(point):
                    pname = f"Product_{j+1}"
                    if not inventory_manager.can_produce(pname, 1):
                        is_feasible = False
                        break
                if is_feasible:
                    feasible_points.append(point)
                attempts += 1

            grid_rand = np.array(feasible_points)
            print(f"   Generated {len(grid_rand)} feasible random combos after {attempts} attempts.")

            if len(grid_rand) > 0:
                results_dict = run_pipeline(grid_rand, bounds, shared_scenarios, J, label="Random-Method")
            else:
                print(f"[WARNING] No feasible combinations found for J={J}")
                results_dict = {"best_profit_enum": 0, "real_profit_opt_poly": 0, "real_profit_opt_nn": 0}

        else:
            if len(bounds) < J:
                print(f"\n[WARNING] Insufficient bounds for J={J}, skipping...")
                results_dict = {"best_profit_enum": 0, "real_profit_opt_poly": 0, "real_profit_opt_nn": 0}
            else:
                num_samples_ = max(200, 100 * J)  # Increase samples for higher dimensions
                max_attempts = num_samples_ * 10

                feasible_points = []
                attempts = 0
                while len(feasible_points) < num_samples_ and attempts < max_attempts:
                    point = random_price_points(J, 1, bounds)[0]
                    if point.size == 0:
                        break

                    is_feasible = True
                    for j, price in enumerate(point):
                        pname = f"Product_{j+1}"
                        if not inventory_manager.can_produce(pname, 1):
                            is_feasible = False
                            break
                    if is_feasible:
                        feasible_points.append(point)
                    attempts += 1

                grid_rand = np.array(feasible_points)
                if len(grid_rand) > 0:
                    print(f"\n[MAIN] Generated {len(grid_rand)} feasible random combos for {J}D after {attempts} attempts.")
                    results_dict = run_pipeline(grid_rand, bounds, shared_scenarios, J, label="Random-Method")
                else:
                    print(f"\n[WARNING] No feasible combinations found for J={J}")
                    results_dict = {"best_profit_enum": 0, "real_profit_opt_poly": 0, "real_profit_opt_nn": 0}

        # Store best results
        best_enum_profits.append(results_dict["best_profit_enum"])
        best_opt_profits_poly.append(results_dict["real_profit_opt_poly"])
        best_opt_profits_nn.append(results_dict["real_profit_opt_nn"])

        # Measure and store runtime
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        print(f"\nRuntime for {J} products: {elapsed:.2f} seconds")

        # -------------------------------------------------
        # Print all runtimes before plotting
        # -------------------------------------------------
        print("\n=== Runtimes Summary ===")
        for J_val, rt in zip(product_list, runtimes):
            print(f"  J={J_val}: {rt:.2f} seconds")

        print("\n=== Profit Summary ===")
        for J_val, enum_p, poly_p, nn_p in zip(product_list, best_enum_profits, best_opt_profits_poly, best_opt_profits_nn):
            print(f"J={J_val}:")
            print(f"  Enumerated/Sampled: {enum_p:.2f}")
            if enum_p != 0:
                poly_improvement = f"({100*(poly_p/enum_p - 1):.1f}% improvement)"
                nn_improvement = f"({100*(nn_p/enum_p - 1):.1f}% improvement)"
            else:
                poly_improvement = "(baseline was 0)"
                nn_improvement = "(baseline was 0)"
            print(f"  Polynomial Opt:     {poly_p:.2f} {poly_improvement}")
            print(f"  Neural Net Opt:     {nn_p:.2f} {nn_improvement}")

    # -------------------------------------------------
    # Scalability: plot runtime
    # -------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(product_list, runtimes, marker='o')
    plt.title("Scalability: Runtime vs. Number of Products")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------
    # Profit Comparison
    # -------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(product_list, best_enum_profits, marker='o', label="Best enumerated/sampled")
    plt.plot(product_list, best_opt_profits_poly, marker='^', label="Optimized (Poly)")
    plt.plot(product_list, best_opt_profits_nn, marker='x', label="Optimized (NN)")
    plt.title("Profit Comparison vs. Number of Products")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------------------------
    # Percentage Increase & Net Change
    # -------------------------------------------------
    pct_increase_poly = []
    pct_increase_nn   = []
    net_change_poly   = []
    net_change_nn     = []

    for enum_profit, poly_profit, nn_profit in zip(best_enum_profits, best_opt_profits_poly, best_opt_profits_nn):
        diff_poly = poly_profit - enum_profit
        net_change_poly.append(diff_poly)
        pct_poly = 100.0 * diff_poly / enum_profit if enum_profit != 0 else (0.0 if diff_poly == 0 else float('inf'))
        pct_increase_poly.append(pct_poly)

        diff_nn = nn_profit - enum_profit
        net_change_nn.append(diff_nn)
        pct_nn = 100.0 * diff_nn / enum_profit if enum_profit != 0 else (0.0 if diff_nn == 0 else float('inf'))
        pct_increase_nn.append(pct_nn)

    plt.figure(figsize=(7,5))
    plt.plot(product_list, pct_increase_poly, marker='^', label='Poly')
    plt.plot(product_list, pct_increase_nn, marker='x', label='NN')
    plt.title("Optimized Profit Percentage Increase vs. Baseline")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Percentage Increase (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(product_list, net_change_poly, marker='^', label='Poly')
    plt.plot(product_list, net_change_nn, marker='x', label='NN')
    plt.title("Net Change in Profit vs. Baseline")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Net Profit Increase")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDone.")

