import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D plotting works
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
import time

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
    def generate_stochastic_demand_linear(prices, num_samples=1000, noise_std=5, scenarios=None):
        """
        Linear demand model + scenario effect. Return DF of shape (num_samples, #products).

        The 'scenarios' parameter is a list of dicts with e.g. {"mean": m, "variance": v}.
        Each sample is drawn from a random scenario from that list, plus noise.
        """
        J = len(prices)
        # Example: base demand and price sensitivity
        base_demands = [100 + 50*i for i in range(J)]
        price_sensitivities = [2 + i for i in range(J)]

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
    def fit_polynomial_surface(revenue_df, poly_degree=3):
        """
        revenue_df has columns [p1, p2, ..., pJ, revenue].
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
    def fit_neural_net_surface(revenue_df,
                               hidden_layer_sizes=(32, 16),
                               max_iter=2000,
                               random_state=42):
        """
        Fit an MLPRegressor (neural net) to the same (X, y) data.
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
    def optimize_prices(response_function, bounds, method='L-BFGS-B', maxiter=500):
        """
        Maximize revenue by minimizing negative of the surrogate model's output.
        """
        def objective(prices):
            val = response_function(*prices)
            # If multiple points, objective is average negative revenue
            if isinstance(val, np.ndarray):
                return -np.mean(val)
            return -val

        init_guess = [(b[0] + b[1])/2 for b in bounds]
        res = minimize(
            objective, init_guess, bounds=bounds, method=method,
            options={'maxiter': maxiter}
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

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
    """
    points = []
    for _ in range(num_samples):
        row = []
        for j in range(J):
            lb, ub = bounds[j]
            val = np.random.uniform(lb, ub)
            row.append(val)
        points.append(row)
    return np.array(points)


# -------------------------------------------------
# Visualization for enumerated data (J=2 or J=3)
# -------------------------------------------------
def visualize_2d_heatmap_enumerated(revenue_df):
    """For J=2 enumerated data, 2D contour/heatmap."""
    if revenue_df.shape[1] != 3:
        return
    p1_unique = sorted(revenue_df["p1"].unique())
    p2_unique = sorted(revenue_df["p2"].unique())
    pivoted   = revenue_df.pivot(index="p2", columns="p1", values="revenue")
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)

    plt.figure(figsize=(8,6))
    cs = plt.contourf(p1_grid, p2_grid, pivoted.values, cmap='viridis', levels=20)
    plt.colorbar(cs)
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.title("Enumerated 2D Heatmap (J=2)")
    plt.show()


def visualize_2d_surface_enumerated(revenue_df):
    """For J=2 enumerated data, 3D surface plot."""
    if revenue_df.shape[1] != 3:
        return
    p1_unique = sorted(revenue_df["p1"].unique())
    p2_unique = sorted(revenue_df["p2"].unique())
    pivoted   = revenue_df.pivot(index="p2", columns="p1", values="revenue")
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)
    Z = pivoted.values

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(p1_grid, p2_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Revenue")
    ax.set_title("Enumerated 3D Surface (J=2)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def visualize_2d_surface_poly(revenue_df, response_func):
    """
    Evaluate a (polynomial or any) response_func in the bounding box of p1/p2,
    then produce 3D surface + 2D contour.
    """
    if revenue_df.shape[1] != 3:
        return
    X_ = revenue_df.iloc[:, :-1]
    p1_vals = np.linspace(X_.iloc[:,0].min(), X_.iloc[:,0].max(), 50)
    p2_vals = np.linspace(X_.iloc[:,1].min(), X_.iloc[:,1].max(), 50)
    p1_grid, p2_grid = np.meshgrid(p1_vals, p2_vals)
    shape_ = p1_grid.shape

    # Evaluate the model
    flattened = np.stack([p1_grid.ravel(), p2_grid.ravel()], axis=-1)
    z_vals    = response_func(*flattened.T).reshape(shape_)

    # 3D surface
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(p1_grid, p2_grid, z_vals, cmap='viridis', edgecolor='none')
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Revenue")
    ax.set_title("Model 3D Surface (J=2)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # 2D contour
    plt.figure(figsize=(10,7))
    cs = plt.contourf(p1_grid, p2_grid, z_vals, cmap='viridis', levels=40)
    plt.colorbar(cs)
    plt.title("Model 2D Contour (J=2)")
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.show()


def visualize_3d_scatter_enumerated(revenue_df):
    """For J=3 enumerated data, 3D scatter w/ color-coded revenue."""
    if revenue_df.shape[1] != 4:
        return
    X1 = revenue_df["p1"].values
    X2 = revenue_df["p2"].values
    X3 = revenue_df["p3"].values
    Y  = revenue_df["revenue"].values

    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(X1, X2, X3, c=Y, cmap='viridis', marker='o', alpha=0.8)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("p3")
    ax.set_title("Enumerated 3D Scatter (J=3)")
    cb = plt.colorbar(sc)
    cb.set_label("Revenue")
    plt.show()


def visualize_3d_scatter_polynomial(revenue_df, response_func, num_points=30):
    """
    Sample bounding box for J=3, show predicted revenue as a 3D scatter.
    We can pass in the polynomial or NN response function.
    """
    if revenue_df.shape[1] != 4:
        return

    p1_min, p1_max = revenue_df["p1"].min(), revenue_df["p1"].max()
    p2_min, p2_max = revenue_df["p2"].min(), revenue_df["p2"].max()
    p3_min, p3_max = revenue_df["p3"].min(), revenue_df["p3"].max()

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
    cb.set_label("Predicted Revenue")
    plt.show()


# -------------------------------------------------
# Main Script
# -------------------------------------------------
if __name__ == "__main__":
    np.random.seed(min(343381, 3434510, 339018, 348306))  # reproducibility

    # Updated product list
    product_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # We'll store results for plotting:
    runtimes = []
    best_enum_revenues = []     # baseline best revenue from enumerated or random approach
    best_opt_revenues_poly = [] # optimized revenue (real) using polynomial model
    best_opt_revenues_nn = []   # optimized revenue (real) using neural network

    # A helper to run the entire pipeline on a set of price combos
    def run_pipeline(price_grid, bounds, scenarios, J, label=""):
        """
        Given a grid of prices (N x J), compute scenario-based revenue,
        fit polynomial + NN, optimize each, and compute real scenario-based revenue
        at the optimized prices. Print results. Return dictionary of relevant stats.
        """
        # 1) Evaluate scenario-based revenue for each price combo
        results_ = []
        for combo_ in price_grid:
            df_ = StochasticOptimization.generate_stochastic_demand_linear(
                prices=combo_,
                num_samples=2000,
                noise_std=10,
                scenarios=scenarios
            )
            rev_ = StochasticOptimization.calculate_revenue(df_, combo_)
            results_.append(list(combo_) + [rev_])

        # Build DataFrame
        col_names_ = [f"p{i+1}" for i in range(J)] + ["revenue"]
        revenue_df_ = pd.DataFrame(results_, columns=col_names_)

        # ----------------------------------------------------------------
        # (Optional) Visualization of enumerated data for J=2 or J=3
        # ----------------------------------------------------------------
        if J == 2:
            visualize_2d_heatmap_enumerated(revenue_df_)
            visualize_2d_surface_enumerated(revenue_df_)
        elif J == 3:
            visualize_3d_scatter_enumerated(revenue_df_)

        # 2) Best enumerated/sampled
        idx_best_ = revenue_df_["revenue"].idxmax()
        best_prices_enum_ = revenue_df_.loc[idx_best_, col_names_[:-1]].values
        best_revenue_enum_ = revenue_df_.loc[idx_best_, "revenue"]
        print(f"({label}) Max revenue (raw combos)    = {best_revenue_enum_:.4f}")
        print(f"({label}) Prices that achieved it:     {best_prices_enum_}")

        # ---------------------------
        # 3) Polynomial Model
        # ---------------------------
        pipe_poly, scY_poly, resp_fn_poly = StochasticOptimization.fit_polynomial_surface(
            revenue_df_, poly_degree=3
        )
        opt_prices_poly, opt_revenue_pred_poly = StochasticOptimization.optimize_prices(
            resp_fn_poly, bounds, method='L-BFGS-B', maxiter=500
        )
        # Evaluate real scenario-based revenue at polynomial-optimized prices
        df_opt_poly = StochasticOptimization.generate_stochastic_demand_linear(
            prices=opt_prices_poly,
            num_samples=2000,
            noise_std=10,
            scenarios=scenarios
        )
        real_revenue_opt_poly = StochasticOptimization.calculate_revenue(df_opt_poly, opt_prices_poly)

        print(f"({label}) [Poly] Predicted revenue = {opt_revenue_pred_poly:.4f}")
        print(f"({label}) [Poly] Real scenario-based revenue  = {real_revenue_opt_poly:.4f}")

        if J == 2:
            visualize_2d_surface_poly(revenue_df_, resp_fn_poly)
        elif J == 3:
            visualize_3d_scatter_polynomial(revenue_df_, resp_fn_poly)

        # ---------------------------
        # 4) Neural Network Model
        # ---------------------------
        mlp_model, scY_nn, resp_fn_nn = StochasticOptimization.fit_neural_net_surface(
            revenue_df_, hidden_layer_sizes=(32,16), max_iter=2000, random_state=42
        )
        opt_prices_nn, opt_revenue_pred_nn = StochasticOptimization.optimize_prices(
            resp_fn_nn, bounds, method='L-BFGS-B', maxiter=500
        )
        # Evaluate real scenario-based revenue at NN-optimized prices
        df_opt_nn = StochasticOptimization.generate_stochastic_demand_linear(
            prices=opt_prices_nn,
            num_samples=2000,
            noise_std=10,
            scenarios=scenarios
        )
        real_revenue_opt_nn = StochasticOptimization.calculate_revenue(df_opt_nn, opt_prices_nn)

        print(f"({label}) [NN]  Predicted revenue = {opt_revenue_pred_nn:.4f}")
        print(f"({label}) [NN]  Real scenario-based revenue  = {real_revenue_opt_nn:.4f}")

        if J == 2:
            # We can reuse the same function for quick 2D contour
            visualize_2d_surface_poly(revenue_df_, resp_fn_nn)
        elif J == 3:
            visualize_3d_scatter_polynomial(revenue_df_, resp_fn_nn)

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
            "best_revenue_enum": best_revenue_enum_,
            "best_prices_enum": best_prices_enum_,
            "opt_prices_poly": opt_prices_poly,
            "real_revenue_opt_poly": real_revenue_opt_poly,
            "opt_prices_nn": opt_prices_nn,
            "real_revenue_opt_nn": real_revenue_opt_nn,
            "revenue_df": revenue_df_
        }

    # -------------------------------------------------
    # Loop over different product counts
    # -------------------------------------------------
    for J in product_list:
        print(f"\n=== Running for {J} product(s) ===")
        start_time = time.time()

        # 1) Generate scenario set for these J products
        print(f"[MAIN] Generating scenario set for {J} products...")
        target_dist = np.random.normal(loc=1.0, scale=0.5, size=1000)
        scenario_means = StochasticOptimization.generate_advanced_scenarios(
            mean=50, variance=100, skewness=0.0, kurtosis=3.0,
            num_scenarios=140, target_distribution=target_dist
        )
        shared_scenarios = [
            {"mean": m, "variance": np.random.uniform(0.8,1.2)}
            for m in scenario_means
        ]

        # 2) Price bounds
        bounds = []
        for i in range(J):
            lb_ = min(20 + 10*i, 100 - 10*i)
            ub_ = max(20 + 10*i, 100 - 10*i)
            bounds.append((lb_, ub_))

        # 3) Decide approach based on J
        if J < 4:
            # For J=2 or J=3, do enumerations
            step = 5  # steps per dimension
            ranges = []
            for (lb_, ub_) in bounds:
                arr_ = np.linspace(lb_, ub_, step)
                ranges.append(arr_)
            grid = np.array(np.meshgrid(*ranges)).T.reshape(-1, J)

            print(f"[MAIN] Enumerating {grid.shape[0]} combos for {J}D grid.")
            results_dict = run_pipeline(grid, bounds, shared_scenarios, J, label="Enumeration")

        elif J == 4:
            # SPECIAL CASE: do BOTH enumerations AND random sampling
            print("[MAIN] J=4 => Doing enumerations first.")
            step = 5
            ranges = []
            for (lb_, ub_) in bounds:
                arr_ = np.linspace(lb_, ub_, step)
                ranges.append(arr_)
            grid_enum = np.array(np.meshgrid(*ranges)).T.reshape(-1, J)
            print(f"   Enumerating {grid_enum.shape[0]} combos.")

            # Enumeration approach
            _ = run_pipeline(grid_enum, bounds, shared_scenarios, J, label="Enum-Method")

            print("\n[MAIN] Now do random sampling approach for J=4.")
            num_samples_ = 42
            grid_rand = random_price_points(J, num_samples_, bounds)
            print(f"   Generated {grid_rand.shape[0]} random combos.")

            # Random approach
            results_dict = run_pipeline(grid_rand, bounds, shared_scenarios, J, label="Random-Method")

        else:
            # For J >= 5, do random sampling approach
            num_samples_ = 42
            print(f"[MAIN] Generating {num_samples_} random combos for {J}D.")
            grid_rand = random_price_points(J, num_samples_, bounds)
            results_dict = run_pipeline(grid_rand, bounds, shared_scenarios, J, label="Random-Method")

        # Store best results
        best_enum_revenues.append(results_dict["best_revenue_enum"])
        best_opt_revenues_poly.append(results_dict["real_revenue_opt_poly"])
        best_opt_revenues_nn.append(results_dict["real_revenue_opt_nn"])

        # Measure and store runtime
        elapsed = time.time() - start_time
        runtimes.append(elapsed)
        # Print the runtime right away
        print(f"Runtime for {J} products: {elapsed:.2f} seconds")

    # -------------------------------------------------
    # Print all runtimes before plotting
    # -------------------------------------------------
    print("\n=== Runtimes Summary ===")
    for J_val, rt in zip(product_list, runtimes):
        print(f"  J={J_val}: {rt:.2f} seconds")

    # -------------------------------------------------
    # Scalability: plot runtime
    # -------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(product_list, runtimes, marker='o', color='b')
    plt.title("Scalability: Runtime vs. Number of Products")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------
    # Revenue Comparison
    # -------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(product_list, best_enum_revenues, marker='o', label="Best enumerated/sampled", color='g')
    plt.plot(product_list, best_opt_revenues_poly, marker='^', label="Optimized (Poly)", color='r')
    plt.plot(product_list, best_opt_revenues_nn,   marker='x', label="Optimized (NN)",   color='b')
    plt.title("Revenue Comparison vs. Number of Products")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Revenue")
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

    for enum_rev, poly_rev, nn_rev in zip(best_enum_revenues, best_opt_revenues_poly, best_opt_revenues_nn):
        # Poly
        diff_poly = poly_rev - enum_rev
        net_change_poly.append(diff_poly)
        pct_poly = 100.0 * diff_poly / enum_rev if enum_rev != 0 else 0.0
        pct_increase_poly.append(pct_poly)

        # NN
        diff_nn = nn_rev - enum_rev
        net_change_nn.append(diff_nn)
        pct_nn  = 100.0 * diff_nn / enum_rev if enum_rev != 0 else 0.0
        pct_increase_nn.append(pct_nn)

    # Plot percentage increase
    plt.figure(figsize=(7,5))
    plt.plot(product_list, pct_increase_poly, marker='^', color='r', label='Poly')
    plt.plot(product_list, pct_increase_nn,   marker='x', color='b', label='NN')
    plt.title("Optimized Revenue Percentage Increase vs. Baseline")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Percentage Increase (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot net change
    plt.figure(figsize=(7,5))
    plt.plot(product_list, net_change_poly, marker='^', color='r', label='Poly')
    plt.plot(product_list, net_change_nn,   marker='x', color='b', label='NN')
    plt.title("Net Change in Revenue vs. Baseline")
    plt.xlabel("Number of Products (J)")
    plt.ylabel("Net Revenue Increase")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDone.")
    
