import seaborn as sns
import shap
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from libraries.config_manager import ConfigManager
from libraries.constants import Base, ConfigParams


def save_fig(image_path, **kwargs):
    plt.savefig(image_path, format='pdf', **kwargs)
    print(f"Saved figure {image_path}")
    plt.close()


def save_fig_with_file_name(image_name, **kwargs):
    config_manager = ConfigManager()
    module_name = config_manager.get_config_parameter(ConfigParams.MODEL_NAME)

    save_path = os.path.join(Base.BASE_PLOT_PATH, module_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_path = os.path.join(save_path, image_name)
    save_fig(image_path, **kwargs)


def plot_shap(X_test, rf_regressor):
    indices = np.random.choice(X_test.shape[0], size=100, replace=False)  # Select random indices
    sample_data = X_test.loc[indices]
    explainer = shap.Explainer(rf_regressor)
    shap_values = explainer.shap_values(sample_data)

    shap.summary_plot(shap_values, sample_data, show=False, )
    save_fig_with_file_name('summary_plot.pdf', bbox_inches='tight')


def plot_PCA(X, y, plot_image_file):
    # Flatten X to 2D array
    X_flat = X.reshape(X.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    # Create scatter plot of X_pca with color-coded labels
    plt.figure(figsize=(8, 6))
    for class_label in np.unique(y):
        indices = np.where(y == class_label)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Class {class_label}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: X vs y')

    plt.legend()
    plt.tight_layout()
    save_fig(plot_image_file)


def plot_distribution(X, y, plot_image_file):
    # Create subplots for each class
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
    fig.suptitle("Distribution of X_train across Classifications", fontsize=16, y=0.92)

    # Select the class labels for plotting
    class_labels = [0, 1, 2]

    # Iterate over the class labels and plot the distribution of X_train for each class
    for i, class_label in enumerate(class_labels):
        # Select the samples for the current class
        samples = X[y == class_label]

        # Flatten the samples for plotting
        samples_flat = samples.flatten()

        # Create a list of labels for the box plot
        labels = [str(class_label)]

        # Create a list of all samples from other classes
        other_samples = X[y != class_label]
        other_samples_flat = other_samples.flatten()

        # Append labels for other classes
        labels.append("Others")

        # Combine the current class samples and samples from other classes
        data = [samples_flat, other_samples_flat]

        # Plot the box plot
        ax = axes[i]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"Class {class_label}")
        ax.set_ylabel("Value")

    plt.tight_layout()
    save_fig(plot_image_file)


def scatter_plot(y_pred, y_test):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot : Actual vs. Predicted')
    save_fig_with_file_name('scatter_plot.pdf')


def residual_plot(y_pred, y_test, outlier_threshold):
    residuals = y_test.values.ravel() - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at 0
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    sns.regplot(x=y_test, y=residuals, scatter=False, lowess=True, line_kws={'color': 'g'}, label='Lowess Smoother')

    # Highlight outliers (customize this part based on your criteria for identifying outliers)
    outliers = np.abs(residuals) > outlier_threshold
    plt.scatter(y_pred[outliers], residuals[outliers], color='r', label='Outliers', marker='x', s=50)
    plt.legend()

    plt.title("Enhanced Residual Plot")
    plt.grid(True)
    plt.show()
    save_fig_with_file_name('residual_plot.pdf')
    return residuals


def plot_hist(data, hist_name, bins=50):
    plt.hist(data, bins=50, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {hist_name}')
    plt.xlabel(f'{hist_name} Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    save_fig_with_file_name(f'hist_{hist_name}_plot.pdf')


def box_plot_crps(crps_value, labels, name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(crps_value, labels=labels, vert=True, patch_artist=True)
    plt.title('Box Plot of CRPS Values')
    plt.xlabel('Category')
    plt.ylabel('CRPS')
    plt.grid(True)
    save_fig_with_file_name(f'box_{name}_qrf.pdf')


def plot_pinball_loss(pinball_loss):
    X_new = list(pinball_loss.keys())
    Y_new = list(pinball_loss.values())

    plt.figure(figsize=(10, 6))
    plt.plot(X_new, Y_new, marker='o', color='royalblue', linestyle='-', linewidth=2, markersize=8)
    plt.fill_between(X_new, Y_new, color='lightblue', alpha=0.5)

    # Set the y-axis to start from 0 and auto-adjust the upper limit based on data
    plt.ylim(0, max(Y_new) * 1.1)

    # Add title and labels
    plt.title('Pinball Loss Across Different Quantiles')
    plt.xlabel('Quantiles')
    plt.ylabel('Pinball Loss')

    # Show the plot
    save_fig_with_file_name(f'pinball_loss.pdf')
