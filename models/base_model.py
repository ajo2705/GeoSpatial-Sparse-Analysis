from abc import abstractmethod, ABC

from libraries.config_manager import ConfigManager
from libraries.constants import ConfigParams


class BaseModel(ABC):
    HYPERPARAMS = []

    def __init__(self):
        self.config_manager = ConfigManager()

    def get_configurations_for_hyper_training(self):
        parameters = {}
        for param in self.HYPERPARAMS:
            parameters[param] = self.config_manager.get_config_parameter(param)

        return parameters

    def get_trial_splits(self, trial, parameter_name, parameter_data):
        limits = parameter_data.get('limits')

        if parameter_data.get('type') == 'int':
            return trial.suggest_int(parameter_name, limits[0], limits[1])

        elif parameter_data.get('type') == 'float':
            return trial.suggest_float(parameter_name, limits[0], limits[1])

    @abstractmethod
    def create_model(self, args):
       pass