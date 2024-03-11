import os
import h5py
import pandas as pd

from constants import Trainer, Base, ConfigParams

COLUMN_NAME = 'columns'


def load_trains(base_path, load_set=Trainer.TRAIN_DATA):
    file_path = os.path.join(base_path, Trainer.TEMPLATE_DATA_PATH.format(filename=load_set))
    print(f"Loading {load_set} from {file_path}")
    with h5py.File(file_path, "r") as hf:
        x = hf["X"][:]
        x_cols = hf["X"].attrs[COLUMN_NAME]
        y = hf["Y"][:]
        y_cols = hf["Y"].attrs[COLUMN_NAME]

        x = pd.DataFrame(x, columns=x_cols)
        y = pd.DataFrame(y, columns=y_cols)
    return x, y


def load_trains_CNN(base_path, load_set=Trainer.TRAIN_DATA):
    file_path = os.path.join(base_path, Trainer.TEMPLATE_DATA_PATH.format(filename=load_set+"_CNN"))
    print(f"Loading {load_set} from {file_path}")
    with h5py.File(file_path, "r") as hf:
        x = hf["X"][:]
        y = hf["Y"][:]

    return x, y


def load_train_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_category_data(category):
    category_str = Trainer.TRAIN_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_TRAINING_DATA_PATH, category_str)


def load_test_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_category_data(category):
    category_str = Trainer.TEST_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_TRAINING_DATA_PATH, category_str)


def load_validation_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_category_data(category):
    category_str = Trainer.VALIDATION_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_RESOURCE_PATH, category_str)


def load_train_linear_data_CNN():
    return load_trains_CNN(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_validation_linear_data_CNN():
    return load_trains_CNN(Base.LINEAR_TRAINING_DATA_PATH, Trainer.VALIDATION_DATA)


def load_test_linear_data_CNN():
    return load_trains_CNN(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TEST_DATA)