from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil

def plot_euclidean_errors(W_train, W_test=None, top_n=7):
    """
    Plot all Euclidean errors for lumisections, highlighting errors above train mean + 2 std.
    """
    latent_space_center = np.mean(W_train, axis=0)

    W_t = W_test if W_test is not None else W_train

    errors = np.linalg.norm(W_t - latent_space_center, axis=1)

    train_errors = np.linalg.norm(W_train - latent_space_center, axis=1)
    train_mean = np.mean(train_errors)
    train_std = np.std(train_errors)
    train_threshold = train_mean + 2 * train_std

    outlier_indices = np.where(errors > train_threshold)[0]
    top_indices = np.argsort(errors)[-top_n:][::-1]

    plt.figure(figsize=(18, 6))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', label="Error per Lumisection (LS)",
             color='orange' if W_test is not None else 'blue')
    plt.xlabel("Lumisection (LS)")
    plt.ylabel("Error (Euclidean Distance)")
    plt.title("Errors for Each Lumisection")

    plt.axhline(y=train_threshold, color='green', linestyle='--', label=f"Train Mean + 2 Std ({train_threshold:.3f})")

    plt.scatter(outlier_indices + 1, errors[outlier_indices], color='red', label="Outliers", s=100)

    for idx in top_indices:
        plt.annotate(f"LS {idx + 1}",
                     (idx + 1, errors[idx]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=9,
                     color='red')
        plt.scatter(idx + 1, errors[idx], color='red', s=100)

    legend_elements = [
        Line2D([0], [0], color='blue', lw=0, label=f"Train Mean: {train_mean:.3f}, Train Std: {train_std:.3f}")
    ]
    if W_test is not None:
        test_mean = np.mean(errors)
        test_std = np.std(errors)
        legend_elements.append(Line2D([0], [0], color='orange', lw=0, label=f"Test Mean: {test_mean:.3f}, Test Std: {test_std:.3f}"))

    for idx in top_indices:
        legend_elements.append(Line2D([0], [0], color='red', marker='o', lw=0, label=f"LS {idx + 1}: Error {errors[idx]:.3f}"))

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(legend_elements)
    labels.extend([elem.get_label() for elem in legend_elements])
    plt.legend(handles, labels, loc="upper left", fontsize="large", bbox_to_anchor=(1.05, 1))
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def calculate_outlier_dataframe(W_train, W_test=None):
    """
    Calculate Euclidean errors for LSs and return a DataFrame with errors
    exceeding train mean + 2 std.
    """
    latent_space_center = np.mean(W_train, axis=0)

    W_t = W_test if W_test is not None else W_train

    errors = np.linalg.norm(W_t - latent_space_center, axis=1)

    train_errors = np.linalg.norm(W_train - latent_space_center, axis=1)
    train_mean = np.mean(train_errors)
    train_std = np.std(train_errors)
    train_threshold = train_mean + 2 * train_std

    df = pd.DataFrame({
        "LS": np.arange(1, len(errors) + 1),
        "Error": errors
    })

    outlier_df = df[df["Error"] > train_threshold].reset_index(drop=True)

    return outlier_df

def calculate_and_plot_mse(matrix, W, H, W_test=None, top_n=7, plot_title="MSE for Each Lumisection"):
    """
    Calculate and plot MSE for each lumisection for training and/or test data.
    """
    mse_errors = []

    W_to_use = W if W_test is None else W_test

    for ls in range(matrix.shape[0]):
        input_data = matrix[ls]
        reconstructed_data = np.dot(W_to_use[ls, :], H)
        mse = np.mean((input_data - reconstructed_data) ** 2)
        mse_errors.append(mse)

    mse_errors = np.array(mse_errors)

    if W_test is None: 
        sorted_indices = np.argsort(mse_errors)
        one_percent_count = ceil(len(mse_errors) * 0.01)
        calculate_and_plot_mse.train_limit = mse_errors[sorted_indices[-(one_percent_count + 1)]] \
            if len(sorted_indices) > one_percent_count else mse_errors.min()

    top_n = min(top_n, len(mse_errors))

    top_indices = np.argsort(mse_errors)[-top_n:][::-1]

    plt.figure(figsize=(18, 6))
    plt.plot(range(1, len(mse_errors) + 1), mse_errors, marker='o', label="MSE per LS")
    plt.xlabel("Lumisection (LS)")
    plt.ylabel("MSE (Mean Squared Error)")
    plt.title(plot_title)

    if hasattr(calculate_and_plot_mse, "train_limit"):
        plt.axhline(y=calculate_and_plot_mse.train_limit, color='green', linestyle='--',
                    label=f"Train Limit: {calculate_and_plot_mse.train_limit:.3e}")

    for idx in top_indices:
        plt.annotate(f"LS {idx + 1}",
                     (idx + 1, mse_errors[idx]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=9,
                     color='red')
        plt.scatter(idx + 1, mse_errors[idx], color='red', s=100)

    legend_labels = [f"LS {idx + 1}: MSE {mse_errors[idx]:.3e}" for idx in top_indices]
    for label in legend_labels:
        plt.scatter([], [], color='red', label=label)

    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()

    return mse_errors

def get_outliers_above_mse_limit(matrix, W, W_test, H):
    """
    Generate a DataFrame of lumisections with MSE values exceeding the train limit for test data.
    """
    if not hasattr(calculate_and_plot_mse, "train_limit"):
        raise ValueError("Train limit has not been calculated yet. Run calculate_and_plot_mse on training data first.")

    train_limit = calculate_and_plot_mse.train_limit 

    mse_errors = []

    for ls in range(matrix.shape[0]):
        input_data = matrix[ls]
        reconstructed_data = np.dot(W_test[ls, :], H)
        mse = np.mean((input_data - reconstructed_data) ** 2)
        mse_errors.append(mse)

    mse_errors = np.array(mse_errors)

    df = pd.DataFrame({
        "LS": np.arange(1, len(mse_errors) + 1),
        "MSE": mse_errors
    })

    outlier_df = df[df["MSE"] > train_limit].reset_index(drop=True)

    return outlier_df

def combine_outlier_dataframes(plot_trend_df, euclid_error_df, mse_limit_df):
    """
    Combine the three dataframes into a single Boolean dataframe with merged LS and respective columns.
    """
    euclid_error_df = euclid_error_df.assign(Euclidean_dist=True)[["LS", "Euclidean_dist"]]
    mse_limit_df = mse_limit_df.assign(MSE=True)[["LS", "MSE"]]

    combined_df = pd.merge(plot_trend_df, euclid_error_df, on="LS", how="outer")
    combined_df = pd.merge(combined_df, mse_limit_df, on="LS", how="outer")

    component_columns = [col for col in plot_trend_df.columns if col != "LS"]
    for column in component_columns:
        if column not in combined_df.columns:
            combined_df[column] = False

    combined_df["Euclidean_dist"] = combined_df["Euclidean_dist"].fillna(False).astype(bool)
    combined_df["MSE"] = combined_df["MSE"].fillna(False).astype(bool)

    return combined_df
