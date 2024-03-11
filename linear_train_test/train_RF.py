"""
Random Forest Linear Regression
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

import numpy as np

from libraries.load_data import load_train_linear_data, load_test_linear_data
from libraries.plot_util import plot_shap, scatter_plot, residual_plot, plot_hist


def train_test():
    X, Y = load_train_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=12)
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = tuple(map(lambda ind: Y.iloc[ind].values.ravel(), [train_index, test_index]))

        rf_regressor.fit(X_train, y_train)
        y_pred = rf_regressor.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Calculate and print the mean MSE and R2 scores across all folds
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    print(f"Mean MSE across {num_folds} folds: {mean_mse:.2f}")
    print(f"Mean R2 across {num_folds} folds: {mean_r2:.2f}")

    X_test, y_test = load_test_linear_data()
    y_pred = rf_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 score : {r2}")
    print(f"MSE score: {mse}")
    print(f"MAE score: {mean_absolute_error(y_test, y_pred)}")

    # SHAP plotter
    plot_shap(X_test, rf_regressor)

    # Scatter Plot
    scatter_plot(y_pred, y_test)

    # Residual plot
    threshold = 20
    residuals = residual_plot(y_pred, y_test, threshold)

    # Histogram of residuals
    plot_hist(residuals, 'Residuals')

    # Histogram of y_test
    plot_hist(y_test, 'y_test')

