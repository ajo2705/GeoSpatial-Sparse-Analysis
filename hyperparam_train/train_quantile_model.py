import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from libraries.load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data
from libraries.metrics import get_pinball_losses_across_quantiles


def train_quantile(quant_model):
    X, Y = load_train_linear_data()
    X_test, y_test = load_test_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_model_mapper = {}
    y_pred_percentiles = []
    net_mse = []
    for quantile in quantiles:
        mse_scores = []
        r2_scores = []

        for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            quant_model.fit(X_train, y_train)
            y_pred = quant_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

        mean_mse = np.mean(mse_scores)
        mean_r2 = np.mean(r2_scores)
        print(f"Mean MSE across {num_folds} folds on quantile {quantile}: {mean_mse:.2f}")
        print(f"Mean R2 across {num_folds} folds on quantile {quantile}: {mean_r2:.2f}")

        quantile_model_mapper[quantile] = quant_model

        y_pred = quant_model.predict(X_test)
        y_pred_percentiles.append(y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        net_mse.append(mse)
        print(f"R2 score on test {quantile} : {r2}")
        print(f"MSE error on test {quantile} : {mse}")

    losses = get_pinball_losses_across_quantiles(quantile_model_mapper, X_test, y_test)
    return -np.mean(net_mse)
