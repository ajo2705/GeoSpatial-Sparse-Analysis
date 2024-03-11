from quantile_forest import RandomForestQuantileRegressor

from libraries.constants import ConfigParams
from models.base_model import BaseModel


class QRFModel(BaseModel):
    HYPERPARAMS = [ConfigParams.N_ESTIMATORS,
                   ConfigParams.DEPTH,
                   ConfigParams.MIN_SAMPLE_LIST,
                   ConfigParams.MIN_SAMPLE_LEAF]

    def __init__(self):
        super().__init__()
        self.isQuantile = True

    def create_model(self, args):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = args
        model = RandomForestQuantileRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=12)
        return model

