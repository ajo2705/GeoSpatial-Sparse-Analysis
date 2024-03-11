import os

import numpy as np
import matplotlib.pyplot as plt
from libraries.constants import Base


def plot_data_distribution(df_X, df_Y):
    num_features = len(df_X.columns)
    num_rows = (num_features // 5) + 1  # Adjust the number of columns as needed

    if not os.path.exists(Base.DISTRIBUTION_PLOT_PATH):
        os.makedirs(Base.DISTRIBUTION_PLOT_PATH)

    for i, column in enumerate(df_X.columns):
        plt.figure(figsize=(6,4))

        plt.hist2d(df_X[column], df_Y.values.flatten(), bins=(20, 20))  # Adjust the number of bins as needed
        plt.title(f'{column} vs Y')
        plt.xlabel(column)
        plt.ylabel('Y')

        plt.ylim([0, 2])

        # Add a color bar
        cbar = plt.colorbar()
        cbar.set_label('Frequency')  # You can customize the label as needed

        save_path = os.path.join(Base.DISTRIBUTION_PLOT_PATH, f'{column}_vs_Y_2D_hist.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the current subplot

        plt.hist(df_X[column], bins=20, alpha=0.5, label="X")
        plt.title(f'{column} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        save_path = os.path.join(Base.DISTRIBUTION_PLOT_PATH, f'{column}.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the current subplot

    plt.figure(figsize=(6, 4))
    plt.hist(df_Y, bins=100, alpha=0.5,
             label='Y (Normalized)')  # Adjust the number of bins as needed
    plt.title(f'Y_vals')
    plt.xlabel(f'Target')
    plt.ylabel('Frequency')
    plt.legend()

    save_path = os.path.join(Base.DISTRIBUTION_PLOT_PATH, f'Y.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the current subplot

    offset = 1e-10
    log_y = np.log10(df_Y + offset)
    plt.figure(figsize=(6, 4))
    plt.hist(log_y, bins=100, alpha=0.5,
             label='Y (Log Normalized)')  # Adjust the number of bins as needed
    plt.title(f'Y_vals')
    plt.xlabel(f'Target')
    plt.ylabel('Frequency')
    plt.legend()

    save_path = os.path.join(Base.DISTRIBUTION_PLOT_PATH, f'Y_log.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the current subplot

    print(f"Plots saved successfully at {Base.DISTRIBUTION_PLOT_PATH}")



