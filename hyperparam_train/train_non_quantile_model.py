import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from libraries.load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data


def train_non_quantile(reg_model):
    X, Y = load_train_linear_data()
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None)

    mse_scores = []
    r2_scores = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(r2_scores)
    print(f"Mean MSE across {num_folds} folds: {mean_mse:.2f}")
    print(f"Mean R2 across {num_folds} folds: {mean_r2:.2f}")

    X_test, y_test = load_test_linear_data()
    y_pred = reg_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 score : {r2}")

    return r2