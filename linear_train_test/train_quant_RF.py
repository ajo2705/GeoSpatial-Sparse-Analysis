"""
Quantile Random Forest on Linear Regression
"""

import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from quantile_forest import RandomForestQuantileRegressor
from libraries.load_data import load_train_linear_data, load_test_linear_data
from libraries.metrics import get_pinball_losses_across_quantiles_x, crps
from libraries.plot_util import plot_hist, box_plot_crps, plot_shap, plot_pinball_loss


def train_test():
    X, Y = load_train_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_model_mapper = {}

    quantile_reg_forest = RandomForestQuantileRegressor(n_estimators=100, default_quantiles=quantiles,
                                                        criterion="squared_error")
    mse_scores = defaultdict(list)
    mae_scores = defaultdict(list)
    r2_scores = defaultdict(list)
    predicted_cdfs = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = tuple(map(lambda ind: Y.iloc[ind].values.ravel(), [train_index, test_index]))

        quantile_reg_forest.fit(X_train, y_train)
        y_pred = quantile_reg_forest.predict(X_test)

        mse = [mean_squared_error(y_test, y_pred[:, i]) for i in range(len(quantiles))]
        mae = [mean_absolute_error(y_test, y_pred[:, i]) for i in range(len(quantiles))]
        r2 = [r2_score(y_test, y_pred[:, i]) for i in range(len(quantiles))]

        for i, quantile in enumerate(quantiles):
            mse_scores[quantile].append(mse[i])
            mae_scores[quantile].append(mae[i])
            r2_scores[quantile].append(r2[i])

    x_test, y_test = load_test_linear_data()
    for quantile in quantiles:
        mean_mse = np.mean(mse_scores[quantile])
        mean_mae = np.mean(mae_scores[quantile])
        mean_r2 = np.mean(r2_scores[quantile])

        print(f"Mean MSE across {num_folds} folds on quantile {quantile}: {mean_mse:.2f}")
        print(f"Mean MAE across {num_folds} folds on quantile {quantile}: {mean_mae:.2f}")
        print(f"Mean R2 across {num_folds} folds on quantile {quantile}: {mean_r2:.2f}")

        quantile_model_mapper[quantile] = quantile_reg_forest
        y_pred = quantile_reg_forest.predict(x_test, quantiles=quantile)
        predicted_cdfs.append(y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 score on test {quantile} : {r2}")
        print(f"MAE score on test {quantile} : {mean_absolute_error(y_test, y_pred)}")
        print(f"MSE score on test {quantile} : {mean_squared_error(y_test, y_pred)}")

    predicted_cdfs = np.transpose(predicted_cdfs)

    kwargs = {"quantiles": quantiles}
    pinball_loss = get_pinball_losses_across_quantiles_x(quantile_model_mapper, x_test, y_test.values.ravel(), kwargs)
    plot_pinball_loss(pinball_loss)

    crps_values = [crps(predicted_cdf, observed_value) for predicted_cdf, observed_value in zip(predicted_cdfs, y_test.values.ravel())]
    # Filter
    prefiltered_len = len(crps_values)
    threshold = 7  # Define your threshold value  --> outliers around 30
    crps_values = [value for value in crps_values if value < threshold]
    print(f"Number of crps scores filtered :{prefiltered_len - len(crps_values)}")

    log_transformed_data = np.log1p(crps_values)

    # Histogram of CRPS values
    plot_hist(crps_values, "crps_QRF", bins=threshold*5)
    plot_hist(log_transformed_data, "crps_log_QRF", bins=threshold * 5)

    # BoxPlot
    crps_value = [crps_values, log_transformed_data]
    labels = ['QRF', 'QRF_log']
    box_plot_crps(crps_value, labels, "QRF")

    plot_shap(x_test, quantile_reg_forest)
