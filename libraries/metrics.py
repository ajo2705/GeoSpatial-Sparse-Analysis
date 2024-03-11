import numpy as np
from sklearn.metrics import make_scorer
from scipy.stats import norm


def pinball_loss(y_true, y_pred, quantile):
    loss = np.mean(np.where(y_true <= y_pred, quantile * (y_pred - y_true), (1 - quantile) * (y_true - y_pred)))
    return loss


def get_pinball_losses_across_quantiles(quantile_model_mapper: dict, X_test, y_test):
    # pinball_scorer = make_scorer(pinball_loss, greater_is_better=False, quantile=quantiles)
    predictions = {quantile: model.predict(X_test) for quantile, model in quantile_model_mapper.items()}

    # Calculate Pinball Loss for each quantile
    pinball_losses = {quantile: pinball_loss(y_test, predictions[quantile], quantile) for quantile in quantile_model_mapper.keys()}

    for quantile in quantile_model_mapper.keys():
        print(f"Pinball Loss ({quantile}): {pinball_losses[quantile]:.2f}")

    return pinball_losses


def get_pinball_losses_across_quantiles_x(quantile_model_mapper: dict, X_test, y_test, kwargs):
    # pinball_scorer = make_scorer(pinball_loss, greater_is_better=False, quantile=quantiles)
    predictions = {quantile: model.predict(X_test, quantiles=quantile) for quantile, model in quantile_model_mapper.items()}

    # Calculate Pinball Loss for each quantile
    pinball_losses = {quantile: pinball_loss(y_test, predictions[quantile], quantile) for quantile in quantile_model_mapper.keys()}

    for quantile in quantile_model_mapper.keys():
        print(f"Pinball Loss ({quantile}): {pinball_losses[quantile]:.2f}")

    return pinball_losses


def fit_normal_distribution(perc_pred, quantiles, target):
    normal_params = {}
    # TODO: Store it in a class for later use
    for i, percentile_predictions in enumerate(perc_pred):
        mean, std_dev = norm.fit(percentile_predictions)
        normal_params[quantiles[i]] = (mean, std_dev)

    mean, std = normal_params[target]
    return norm.ppf(target, loc=mean, scale=std)


def crps(F, x):
    """
    Calculate the Continuous Ranked Probability Score (CRPS)

    Parameters:
    F (array-like): Predicted CDF values at different points
    x (float): Observed value

    Returns:
    float: CRPS value
    """
    y_values = np.linspace(start=min(F), stop=max(F), num=len(F))
    H = np.heaviside(y_values - x, 0.5)  # Heaviside step function
    integral = np.trapz((F - H) ** 2, y_values)  # Trapezoidal integration
    return integral
