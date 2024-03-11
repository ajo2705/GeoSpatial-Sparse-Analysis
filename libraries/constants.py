class Trainer:
    TRAIN_DATA = "train"
    TEST_DATA = "test"
    VALIDATION_DATA = "valid"

    TEMPLATE_DATA_PATH = "{filename}_data.h5"


class Base:
    # TODO: Modify BASE_PATH and add it in configuration yml
    BASE_PATH = "./"
    BASE_RESOURCE_PATH = BASE_PATH + "Resources/"
    BASE_IMAGE_PATH = BASE_RESOURCE_PATH + "Images/"
    BASE_XLSX_PATH = BASE_RESOURCE_PATH + "Tables/"
    BASE_RASTER_PATH = BASE_RESOURCE_PATH + "Rasters/"
    BASE_RASTER_DATAFRAME_PATH = BASE_RESOURCE_PATH + "DataFrame/"
    BASE_MODEL_PATH = BASE_RESOURCE_PATH + "Models/"

    BASE_TRAINING_DATA_PATH = BASE_RESOURCE_PATH + "Training/"
    CLASSIFIER_TRAINING_DATA_PATH = BASE_TRAINING_DATA_PATH + "classification_dataset/"
    LINEAR_TRAINING_DATA_PATH = BASE_TRAINING_DATA_PATH + "linear_dataset/"

    BASE_PLOT_PATH = BASE_RESOURCE_PATH + "Plots/"
    DISTRIBUTION_PLOT_PATH = BASE_PLOT_PATH + "Distribution/"

    RASTER_IMAGE_PATH = BASE_RASTER_PATH + "images/"
    TRAINING_IMAGE_PATH = BASE_IMAGE_PATH + "training_data/"

    BASE_LOG_PATH = BASE_PATH + "Logs/"
    TENSOR_LOG_PATH = BASE_LOG_PATH + "tensorLog/"
    HYPERPARAM_LOG_PATH = BASE_LOG_PATH + "hyperparam/tensorLog/"

    BASE_MODEL_STORE_PATH = BASE_MODEL_PATH + "trained_models/"
    BASE_HYPER_MODEL_STORE_PATH = BASE_MODEL_PATH + "hyper_run/"


class ConfigParams:
    CONFIG_FILE = "config.yml"

    # Model parameters
    MODEL_NAME = "model"
    MODEL_CLASS = "class"

    RF_TRAINERS = "random_forest_trainers"
    N_ESTIMATORS = RF_TRAINERS + "/n_estimators"
    DEPTH = RF_TRAINERS + "/max_depth"
    MIN_SAMPLE_LIST = RF_TRAINERS + "/min_samples_list"
    MIN_SAMPLE_LEAF = RF_TRAINERS + "/min_samples_leaf"

    HYPERPARAMETER = "hyperparameter"
    LOG_PATH = HYPERPARAMETER + "/log_path"
    METRIC = HYPERPARAMETER + "/metric"


