import os
import sys
import importlib.util

# Place your base path
BASE_PATH = "./"

HYPERPARAM_LIBRARY = "hyperparam_train"
LIB = "libraries"
RASTERS = "raster_processor"
MODEL_TRAINER = "models"
# LOSS_LANDSCAPE = "loss_landscape"

lib_locations = [os.path.join(BASE_PATH, lib) for lib in [LIB, RASTERS, HYPERPARAM_LIBRARY, MODEL_TRAINER]]
sys.path.extend(lib_locations)

from raster_processor.raster_converter_linear import process_raster
from hyperparam_train.optuna_training import optuna_trainer
from libraries.config_manager import ConfigManager
from libraries.constants import ConfigParams

class_mapper = {"RandomForest": "train_random_forest",
                "QuantileRandomForest": "train_quant_random_forest",
                "CatBoost": "train_cat_boost",
                "CNN": "train_CNN",
                "QKNN": "train_quantile_KNN",
                "QuantileGBR": "train_QGBR",
                "QuantileLR": "train_QLR"}


def train_test():
    config_manager = ConfigManager()
    module_name = config_manager.get_config_parameter(ConfigParams.MODEL_NAME)

    if config_manager.get_config_parameter(ConfigParams.USE_HYPERPARAMETER_TRAINING):
        optuna_trainer()

    file_path = "linear_train_test/" + class_mapper[module_name] + ".py"
    module_name = "train_test"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.train_test()


if __name__ == '__main__':
    train_test()
    # For raster processing
    # process_raster()
