"""
GradientBoostingRegressor on Linear Regression
"""
import numpy as np

from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from libraries.load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data


def train_test():
    X, Y = load_train_linear_data()
    X_test, y_test = load_test_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_model_mapper = {}
    y_pred_percentiles = []

    for quantile in quantiles:
        ngb = NGBRegressor(Dist=Normal, n_estimators=100, learning_rate=0.01)

        mse_scores = []
        r2_scores = []

        for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = tuple(map(lambda ind: Y.iloc[ind].values.ravel(), [train_index, test_index]))

            ngb.fit(X_train, y_train)
            y_pred = ngb.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

        mean_mse = np.mean(mse_scores)
        mean_r2 = np.mean(r2_scores)
        print(f"Mean MSE across {num_folds} folds on quantile {quantile}: {mean_mse:.2f}")
        print(f"Mean R2 across {num_folds} folds on quantile {quantile}: {mean_r2:.2f}")

        quantile_model_mapper[quantile] = ngb

        y_pred = ngb.predict(X_test)
        y_pred_percentiles.append(y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"R2 score on test {quantile} : {r2}")



