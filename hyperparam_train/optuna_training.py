import os
import optuna
import importlib
from optuna.integration import TensorBoardCallback

from libraries.config_manager import ConfigManager
from libraries.constants import ConfigParams, Base
from hyperparam_train.train_non_quantile_model import train_non_quantile
from hyperparam_train.train_quantile_model import train_quantile


def load_model(config_manager):
    module_name = config_manager.get_config_parameter(ConfigParams.MODEL_NAME)
    class_name = config_manager.get_config_parameter(ConfigParams.MODEL_CLASS)

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model = model_class()
    return model


def train_model(model):
    if model.isQuantile:
        train_quantile(model)
    return train_non_quantile(model)


def objective(trial):
    config_manager = ConfigManager()
    hyper_model = load_model(config_manager)

    parameters = hyper_model.get_configurations_for_hyper_training()
    trial_values = []
    for parameter_name, parameter_data in parameters.items():
        trial_values.append(hyper_model.get_trial_splits(trial, parameter_name, parameter_data))

    model = hyper_model.create_model(tuple(trial_values))
    if hyper_model.isQuantile:
        value = train_quantile(model)
    else:
        value = train_non_quantile(model)
    return value


def optuna_trainer():
    config_manager = ConfigManager()
    study = optuna.create_study(direction='maximize')
    log_path = os.path.join(Base.HYPERPARAM_LOG_PATH, config_manager.get_config_parameter(ConfigParams.LOG_PATH))
    tensorboard_callback = TensorBoardCallback(log_path,
                                               metric_name=config_manager.get_config_parameter(ConfigParams.METRIC))

    study.optimize(objective, n_trials=100, callbacks=[tensorboard_callback])





