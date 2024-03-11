"""
CatBoost Linear Regression
"""
from catboost import CatBoostRegressor
from catboost import Pool, cv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from libraries.load_data import load_train_linear_data, load_test_linear_data, load_validation_linear_data
from libraries.plot_util import plot_shap, scatter_plot, residual_plot, plot_hist


def train_test():
    categorical_vars = ['soil_taxonomy', 'CDL_1km', 'LC_type1']
    X, Y = load_train_linear_data()
    val_x, val_y = load_validation_linear_data()

    X[categorical_vars] = X[categorical_vars].astype(str).replace('nan', 'Unknown')
    val_x[categorical_vars] = val_x[categorical_vars].astype(str).replace('nan', 'Unknown')

    train_data = Pool(data=X, label=Y, cat_features=categorical_vars)

    val_data = Pool(data=val_x, label=val_y, cat_features=categorical_vars)

    params = {
        'iterations': 20000,  # Number of boosting iterations
        'depth': 10,  # Depth of the trees
        'learning_rate': 0.15,
        'l2_leaf_reg' : 5.403077451698753,
        'early_stopping_rounds': 1000,
        'loss_function': 'RMSE'  # Use RMSE as the loss function (change if needed)
    }

    # Perform cross-validation
    cv_results = cv(train_data, params, num_boost_round=100, nfold=5, seed=0)
    # Extract and print the mean and standard deviation of RMSE
    mean_rmse = cv_results['test-RMSE-mean'].min()
    std_rmse = cv_results['test-RMSE-std'].min()

    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {std_rmse:.4f}")

    boost_model = CatBoostRegressor(**params)
    boost_model.fit(train_data, eval_set=val_data)

    X_test, y_test = load_test_linear_data()
    X_test[categorical_vars] = X_test[categorical_vars].astype(str).replace('nan', 'Unknown')

    y_pred = boost_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 score : {r2}")
    print(f"MSE score: {mse}")
    print(f"MAE score: {mean_absolute_error(y_test, y_pred)}")

    # SHAP plotter
    plot_shap(X_test, boost_model)

    # Scatter Plot
    scatter_plot(y_pred, y_test)

    # Residual plot
    threshold = 20
    residuals = residual_plot(y_pred, y_test, threshold)

    # Histogram of residuals
    plot_hist(residuals, 'Residuals')

    # Histogram of y_test
    plot_hist(y_test, 'y_test')
