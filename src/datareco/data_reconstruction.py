import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class DataReconstruction:
    def __init__(self, W, H, dense_matrix, W_test=None, dense_matrix_test=None, x_min=0, x_max=80000, model_name="Model"):
        self.W = W
        self.H = H
        self.dense_matrix = dense_matrix
        self.W_test = W_test
        self.dense_matrix_test = dense_matrix_test
        self.x_min = x_min
        self.x_max = x_max
        self.model_name = model_name
        self.x_values = np.linspace(x_min, x_max, dense_matrix.shape[1])

    def calculate_area_contributions(self, ls, W, H):
        """
        Calculate area contributions for each component in a given lumisection.
        """
        total_areas = 0
        areas = []
        for i in range(H.shape[0]):
            component_contribution = W[ls, i] * H[i]
            area = np.trapz(component_contribution)
            areas.append(area)
            total_areas += area

        area_percentages = [(area / total_areas) * 100 for area in areas]
        return area_percentages

    def plot_trend(self, df, title="Component Contribution Trend", components="all", mode="area"):
        """
        Plot trend of component contributions over lumisections by area or coefficient.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        if components == "all":
            columns_to_plot = df.columns[1:] 
        else:
            columns_to_plot = [f"Component_{i}" for i in components]
        
        ylabel = "Area Contribution (%)" if mode == "area" else "Coefficient Contribution (%)"

        ls_values = df['LS'].values
        extended_ls_values = np.append(ls_values, ls_values[-1] + 1)

        for column in columns_to_plot:
            ax.stairs(df[column].values, extended_ls_values, label=column.replace('Component_', 'Component '), baseline=None)

        ax.set_xlim(left=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins='auto'))
        
        ticks = ax.get_xticks()
        if ticks[0] != 1:
            ticks = np.insert(ticks, 0, 1)  
            ax.set_xticks(ticks)
        
        ax.set_title(f"{title} ({mode.capitalize()} Mode)")
        ax.set_xlabel("Lumisection")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()

    def collect_contributions(self, W, H, dense_matrix, num_ls, data_type="train", mode="area"):
        """
        Collect area or coefficient contributions for all lumisections for training or testing data.
        """
        if num_ls == "all":
            num_ls = W.shape[0]
    
        contribution_data = {"LS": []}
        for i in range(H.shape[0]):
            contribution_data[f"Component_{i+1}"] = []

        for ls in range(num_ls):
            contribution_data["LS"].append(ls + 1)

            if mode == "area":
                contribs = self.calculate_area_contributions(ls, W, H)
            elif mode == "coef":
                total_coef = np.sum(W[ls, :])
                contribs = [(W[ls, i] / total_coef) * 100 for i in range(H.shape[0])]
            else:
                raise ValueError("Invalid mode. Choose 'area' or 'coef'.")

            for idx, contrib in enumerate(contribs):
                contribution_data[f"Component_{idx+1}"].append(contrib)

        return pd.DataFrame(contribution_data)
        
    def print_component_contributions(self, ls, W, mode="train"):
        """
        Prints the component contributions for either training or testing data.
        """
        area_contributions = self.calculate_area_contributions(ls, W, self.H)
        total_W_contrib = np.sum(W[ls, :])

        print(f"\n--- {self.model_name} Component Contributions ({mode.capitalize()}) ---")
        for i in range(self.H.shape[0]):
            percent_area_contrib = area_contributions[i]
            percent_w_contrib = (W[ls, i] / total_W_contrib) * 100
            print(f"{self.model_name} Component {i+1} ({mode}) - Area Contribution: {percent_area_contrib:.2f}%, "
                  f"W Contribution: {percent_w_contrib:.2f}%, "
                  f"W Coefficient: {W[ls, i]:.6f}")

    def plot_reconstruction(self, ls_train, ls_test=None):
        """
        Plot reconstruction and component contributions for a given lumisection using stairs plot with area and coefficient contributions.
        """
        fig, axs = plt.subplots(1, 2, figsize=(16, 7)) 

        x_values = np.linspace(self.x_min, self.x_max, self.dense_matrix.shape[1] + 1)

        # Training data plot
        self.print_component_contributions(ls_train, self.W, mode="train")
        train_areas = self.calculate_area_contributions(ls_train, self.W, self.H)
        reconstructed_sample_train = np.dot(self.W[ls_train, :], self.H)
        axs[0].stairs(self.dense_matrix[ls_train], x_values, label='Input train', color="g")

        for i in range(self.H.shape[0]):
            component_contribution_train = self.W[ls_train, i] * self.H[i]
            percent_area_contrib_train = train_areas[i]
            percent_w_contrib_train = (self.W[ls_train, i] / np.sum(self.W[ls_train, :])) * 100
            axs[0].stairs(component_contribution_train, x_values, 
                      label=f'Component {i+1} ({percent_area_contrib_train:.2f}% area, {percent_w_contrib_train:.2f}% W)')

        axs[0].stairs(reconstructed_sample_train, x_values, label='Reconstructed train', linestyle='--', color='r')
        axs[0].set_title(f"Reconstruction of Training Data for LS {ls_train+1} ({self.model_name})")
        axs[0].legend()

        # Testing data plot, if provided
        if self.W_test is not None and self.dense_matrix_test is not None and ls_test is not None:
            self.print_component_contributions(ls_test, self.W_test, mode="test")
            test_areas = self.calculate_area_contributions(ls_test, self.W_test, self.H)
            reconstructed_sample_test = np.dot(self.W_test[ls_test, :], self.H)
            axs[1].stairs(self.dense_matrix_test[ls_test], x_values, label='Input test', color="g")

            for i in range(self.H.shape[0]):
                component_contribution_test = self.W_test[ls_test, i] * self.H[i]
                percent_area_contrib_test = test_areas[i]
                percent_w_contrib_test = (self.W_test[ls_test, i] / np.sum(self.W_test[ls_test, :])) * 100
                axs[1].stairs(component_contribution_test, x_values, 
                          label=f'Component {i+1} ({percent_area_contrib_test:.2f}% area, {percent_w_contrib_test:.2f}% W_test)')

            axs[1].stairs(reconstructed_sample_test, x_values, label='Reconstructed test', linestyle='--', color='r')
            axs[1].set_title(f"Reconstruction of Testing Data for LS {ls_test+1} ({self.model_name})")
            axs[1].legend()

        plt.tight_layout()
        plt.show()
            
    def plot_all_components(self, variable_name='H'):
        """
        Plots each component in matrix H on a single plot using stairs with specified x_min and x_max values.
        """
        n_components, n_features = self.H.shape
        x_values = np.linspace(self.x_min, self.x_max, n_features + 1)

        plt.figure(figsize=(10, 6))
        
        for i in range(n_components):
            plt.stairs(self.H[i, :], x_values, label=f'Component {i+1}', baseline=None)
        
        plt.xlabel('Feature Value (x)')
        plt.ylabel('Component Contribution')
        plt.title(f'Components of {variable_name} Matrix')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()