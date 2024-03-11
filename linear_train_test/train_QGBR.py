"""
Quantile GradientBoostingRegressor on Linear Regression
"""
import numpy as np
from collections import defaultdict

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from libraries.load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data
from libraries.metrics import get_pinball_losses_across_quantiles, crps
from libraries.plot_util import plot_hist, box_plot_crps, plot_shap, plot_pinball_loss


def train_test():
    X,Y = load_train_linear_data()
    X_test, Y_test = load_test_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_model_mapper = {}
    y_pred_percentiles = []
    for quantile in quantiles:
        gbr = GradientBoostingRegressor(n_estimators=100, loss="quantile", alpha=quantile)

        mse_scores = []
        r2_scores = []
        mae_scores = []

        for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = tuple(map(lambda ind: Y.iloc[ind].values.ravel(), [train_index, test_index]))

            gbr.fit(x_train, y_train)
            y_pred = gbr.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

        mean_mse = np.mean(mse_scores)
        mean_r2 = np.mean(r2_scores)
        mean_mae = np.mean(mae_scores)
        print(f"Mean MSE across {num_folds} folds on quantile {quantile}: {mean_mae:.2f}")
        print(f"Mean MAE across {num_folds} folds on quantile {quantile}: {mean_mse:.2f}")
        print(f"Mean R2 across {num_folds} folds on quantile {quantile}: {mean_r2:.2f}")

        quantile_model_mapper[quantile] = gbr

        y_pred = gbr.predict(X_test)
        y_pred_percentiles.append(y_pred)
        r2 = r2_score(Y_test, y_pred)
        mse = mean_squared_error(Y_test, y_pred)
        print(f"R2 score on test {quantile} : {r2}")
        print(f"MSE score on test {quantile} : {mse}")

    pinball_loss = get_pinball_losses_across_quantiles(quantile_model_mapper, X_test, Y_test.values.ravel())
    plot_pinball_loss(pinball_loss)

    predicted_cdfs = np.transpose(y_pred_percentiles)
    crps_values = [crps(predicted_cdf, observed_value) for predicted_cdf, observed_value in
                   zip(predicted_cdfs, Y_test.values.ravel())]

    # Filter
    prefiltered_len = len(crps_values)
    threshold = 7  # Define your threshold value  --> outliers around 30
    crps_values = [value for value in crps_values if value < threshold]
    print(f"Number of crps scores filtered :{prefiltered_len - len(crps_values)}")

    log_transformed_data = np.log1p(crps_values)

    # Histogram of CRPS values
    plot_hist(crps_values, "crps_QKNN", bins=threshold * 5)
    plot_hist(log_transformed_data, "crps_log_QKNN", bins=threshold * 5)

    # BoxPlot
    crps_value = [crps_values, log_transformed_data]
    labels = ['QGBR', 'QGBR_log']
    box_plot_crps(crps_value, labels, "QGBR")

    plot_shap(X_test, quantile_model_mapper[0.5])
