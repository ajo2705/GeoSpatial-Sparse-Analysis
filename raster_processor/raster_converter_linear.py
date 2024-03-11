import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from libraries.constants import Trainer, Base
from libraries.raster_util import store_data_files, raster_stack_to_dataframe, target_raster_to_dataframe, \
    remove_outliers_iqr
from libraries.plot_data_distribution import plot_data_distribution


TRAIN_DATA = Trainer.TRAIN_DATA
TEST_DATA = Trainer.TEST_DATA
VALIDATION_DATA = Trainer.VALIDATION_DATA
IMAGE_FILE_PATH = Base.TRAINING_IMAGE_PATH

TRAIN_DATA_TARGET_FILEPATH = Trainer.TEMPLATE_DATA_PATH
target = "TARGET"

FILTER_CHANNELS = ['elev', 'slope', 'upslope_curvature', 'profile_curvature', 'downslope_curvature',
                   'local_upslope_curvature', 'local_downslope_curvature', 'aspect', 'bio_1', 'bio_2', 'bio_3', 'bio_4',
                   'bio_5', 'bio_8', 'bio_9', 'bio_18', 'NDVI_p0', 'CDL_1km', 'LC_type1', 'EVI_p0', 'Final_sm', 'ai',
                   'et0', 'clay_b0', 'clay_b10', 'clay_b30', 'clay_b60', 'clay_b100', 'sand_b0', 'sand_b10', 'sand_b30',
                   'sand_b60', 'sand_b100', 'srad', 'soil_taxonomy', 'SBIO_8', '5to15cm_SBIO_2', '5to15cm_SBIO_3',
                   'NoData_Mask', 'Latitude', 'Longitude']


def load_process_data(raster_abs_path, target_abs_path):
    co_variate_data, no_data_mask = raster_stack_to_dataframe(raster_abs_path)
    data = co_variate_data.loc[:, FILTER_CHANNELS]

    target_data_df = target_raster_to_dataframe(target_abs_path)
    data[target] = target_data_df[target]

    data.dropna()
    remove_points = data['NoData_Mask'] == 1
    data = data.loc[~remove_points, :].reset_index(drop=True)

    # drop rows with missing soc values
    data = data.dropna(subset=[target])
    data = data[data[target] >= 0]
    # data = data[(data["crop_land"] != '0')].reset_index(drop=True)
    data.drop('NoData_Mask', axis=1)

    remove_outliers_iqr(data, target, q1=0.25, q3=0.75)

    # Get classification labels for one hot encoding
    max_values = data.max()
    selected_columns = max_values[max_values > 1].index.tolist()
    classification_columns = list(set(selected_columns) - {target, 'Longitude', 'Latitude'})

    print(f"Performing one hot encoding on {classification_columns}")
    encoded_df = pd.get_dummies(data, columns=classification_columns)

    y = encoded_df[target]

    epsilon = 1e-10
    adjusted_y = np.where(y == 0, epsilon, y)
    logy = np.log10(adjusted_y)

    x = encoded_df.drop([target, 'Latitude', 'Longitude'], axis=1)

    plot_data_distribution(x, y)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.1, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=0)

    X_train, X_val, X_test = tuple(map(lambda d: pd.DataFrame(d, columns=x.columns), [X_train, X_val, X_test]))
    y_train, y_val, y_test = tuple(map(lambda d: pd.DataFrame(d, columns=[target]), [y_train, y_val, y_test]))

    return {TRAIN_DATA: (X_train, y_train),
            TEST_DATA: (X_test, y_test),
            VALIDATION_DATA: (X_val, y_val)}


def process_raster():
    covariate_rast_path = "CO_VARIATE_RASTER"
    target_rast_path = "TGT_RASTER"

    path = Base.BASE_RASTER_PATH
    raster_abs_path = os.path.join(path, covariate_rast_path)
    target_abs_path = os.path.join(path, target_rast_path)

    data_set = load_process_data(raster_abs_path, target_abs_path)
    path = os.path.join(Base.LINEAR_TRAINING_DATA_PATH)

    store_data_files(data_set, path)

