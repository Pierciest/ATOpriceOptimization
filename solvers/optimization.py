from scipy.optimize import minimize

def optimize_prices(model, scaler_X, scaler_y):
    slope = model.coef_
    intercept = model.intercept_

    response_function = lambda s, d: intercept + slope[0]*s + slope[1]*d + slope[2]*s*d + slope[3]*s**2 + slope[4]*d**2

    objective_function = lambda x: -response_function(x[0], x[1])
    initial_guess = [0, 0]
    bounds = [(-1, 1), (-1, 1)]
    result = minimize(objective_function, initial_guess, bounds=bounds)

    optimal_x1, optimal_x2 = result.x
    optimal_response = -objective_function([optimal_x1, optimal_x2])

    original_scale = scaler_X.inverse_transform([[optimal_x1, optimal_x2, optimal_x1**2, optimal_x2**2, optimal_x1*optimal_x2]])
    optimal_x1_original, optimal_x2_original = original_scale[0][:2]
    optimal_response_original = scaler_y.inverse_transform([[optimal_response]])[0][0]

    return optimal_x1_original, optimal_x2_original, optimal_response_original
