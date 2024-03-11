import os

import rasterio
import h5py
import numpy as np
import pandas as pd
from math import ceil
from enum import Enum
from matplotlib import pyplot as plt

from libraries.constants import Base, Trainer
from libraries.plot_util import save_fig

COLUMN_NAME = 'columns'


def raster_stack_to_dataframe(raster_stack_file):
    with rasterio.open(raster_stack_file) as src:
        stack_data = src.read()
        stack_data = np.transpose(stack_data, (1, 2, 0))
        stack_data = stack_data.reshape(-1, src.count)

        band_names = [src.tags(i)['NAME'] for i in range(1, src.count + 1)]

        nodata_band_idx = band_names.index('NoData_Mask') + 1

    df = pd.DataFrame(stack_data, columns=band_names)

    nodata_mask = df['NoData_Mask'] == 1

    # Compute longitude and latitude columns
    nrows, ncols = src.shape
    rows, cols = np.indices((nrows, ncols))
    x = cols * src.transform.a + src.transform.c
    y = rows * src.transform.e + src.transform.f

    # Add the latitude and longitude columns to the DataFrame
    df['Longitude'] = x.flatten()
    df['Latitude'] = y.flatten()

    return df, nodata_mask


def target_raster_to_dataframe(target_raster_file):
    with rasterio.open(target_raster_file) as src:
        target_data = src.read(1)
        target_data = target_data.flatten()

        df = pd.DataFrame({'TARGET': target_data})

        # Compute longitude and latitude columns
        nrows, ncols = src.shape
        rows, cols = np.indices((nrows, ncols))
        x = cols * src.transform.a + src.transform.c
        y = rows * src.transform.e + src.transform.f

        # Add the latitude and longitude columns to the DataFrame
        df['Longitude'] = x.flatten()
        df['Latitude'] = y.flatten()

    return df


def single_raster_to_dataframe(raster_file, column_name):
    with rasterio.open(raster_file) as src:
        raster_data = src.read(1)
        raster_data = raster_data.flatten()

        df = pd.DataFrame({column_name: raster_data})

    return df


def get_raster_crs(raster_file):
    with rasterio.open(raster_file) as src:
        crs = src.crs
    return crs


# Create a function to detect outliers based on IQR
def detect_outliers_iqr(df, column, q1=0.25, q3=0.75):
    q1_value = df[column].quantile(q1)
    q3_value = df[column].quantile(q3)
    iqr = q3_value - q1_value
    outliers = df[(df[column] < (q1_value - 1.5 * iqr)) | (df[column] > (q3_value + 1.5 * iqr))]
    return outliers


# Remove Outliers
def remove_outliers_iqr(df, column, q1=0.25, q3=0.75):
    outliers = detect_outliers_iqr(df, column, q1, q3)
    df = df[~df.index.isin(outliers.index)]
    return df


def percentile_based_boundary(target_values):
    # More equal distribution -> see how training fares with this
    # 1 is added to max for safety; proper working of digitize
    class_boundaries = np.percentile(target_values, [33, 66])
    class_boundaries = np.concatenate(([0, ceil(np.max(target_values) + 1)], class_boundaries))
    class_boundaries = np.sort(class_boundaries)
    return class_boundaries


def plot_data_non_outliers(non_outlier_targets, bins):
    hist_plot = f"hist_plot_{bins}.jpeg"
    plt.hist(non_outlier_targets, bins=bins)

    hist_plot_image_path = os.path.join(Base.TRAINING_IMAGE_PATH, hist_plot)
    save_fig(hist_plot_image_path)


def store_data_files(data_dict, base_path):
    for key, val in data_dict.items():
        abs_path = _get_file_handle(base_path, key)
        print(f"Storing data in {abs_path}")
        with h5py.File(abs_path, "w") as handle:
            for dataset, data_type in zip(val, ["X", "Y"]):
                handle.create_dataset(data_type, data=dataset)
                handle[data_type].attrs[COLUMN_NAME] = dataset.columns.tolist()


def store_data_files_CNN(data_dict, base_path):
    print(f"Storing at location {base_path}")
    for key, val in data_dict.items():
        abs_path = _get_file_handle(base_path, key + "_CNN")
        with h5py.File(abs_path, "w") as handle:
            for dataset, data_type in zip(val, ["X", "Y"]):
                handle.create_dataset(data_type, data=dataset)


def _get_file_handle(base_path, key):
    file_path = Trainer.TEMPLATE_DATA_PATH.format(filename=key)
    abs_path = os.path.join(base_path, file_path)
    if not (os.path.exists(abs_path)):
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


class BoundaryType(Enum):
    Percentile = 0
    Custom = 1


