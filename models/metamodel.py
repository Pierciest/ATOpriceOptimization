import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

def fit_metamodel(revenue_df, save_path="models/metamodel.pkl"):
    revenue_df['P1^2'] = revenue_df['P1'] ** 2
    revenue_df['P2^2'] = revenue_df['P2'] ** 2
    revenue_df['P1_P2'] = revenue_df['P1'] * revenue_df['P2']
    X = revenue_df[['P1', 'P2', 'P1^2', 'P2^2', 'P1_P2']].values
    y = revenue_df['Revenue'].values

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    model = Ridge(alpha=1e-3)
    model.fit(X_normalized, y_normalized)

    # Save model and scalers
    joblib.dump({"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y}, save_path)
    return model, scaler_X, scaler_y

def load_metamodel(load_path="models/metamodel.pkl"):
    data = joblib.load(load_path)
    return data["model"], data["scaler_X"], data["scaler_y"]
