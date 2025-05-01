import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

import numpy as np
import pandas as pd

import re

from dataclasses import dataclass
from typing import List


from config import PROJECT_ROOT, CONFIG_DIR

from src.utils.helpers import load_data_paths
from src.utils.metrics import test_second_order_stochastic_dominance

# Setup logger
import logging
from src.utils.custom_formatter import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)


@dataclass
class ReturnSeries:
    label: str
    df: pd.DataFrame
    horizons: List[int]
    color: str = None

class Plotter:
    # counter fro number of different "indeces"
    index_counter = 0 

    def __init__(self, series_list: List[ReturnSeries]):
        self.series_list = series_list
        self.palette = sns.color_palette("muted")
        
        data_paths = load_data_paths()
        self.images_dir = PROJECT_ROOT / data_paths["images_dir"]

        self._update_counter()

    def _update_counter(self):
        """
        Update counter for the number of different securties by iterating over each series in series_list
        """
        if len(self.series_list) > 1 :
            for series in self.series_list:
                # get the individual df columns and find base securities
                cols = series.df.columns

                # Obtain base cols (columsn without returns name)
                base_cols = {col for col in cols if not re.search(r"_\d+_return$", col)}
                
                if len(base_cols) == 0:
                    base_cols = cols
                
                Plotter.index_counter += len(base_cols)
        else:
            # get the individual df columns and find base securities
            cols = self.series_list[0].df.columns

            # Obtain base cols (columsn without returns name)
            base_cols = {col for col in cols if not re.search(r"_\d+_return$", col)}
            
            if len(base_cols) == 0:
                base_cols = cols
            
            Plotter.index_counter += len(base_cols)


    def plot_returns_timeseries(self,
                                custom_labels: List[str] = None, 
                                save: bool = False, 
                                file_name: str = "returns_plot.png", 
                                title: str = "Rolling Market vs Retail Returns Across Horizons", 
                                show: bool = True):

        if not self.series_list:
            raise ValueError("No return series provided.")

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Use horizons from the first series (assumed common)
        horizons = self.series_list[0].horizons
        rows = int(np.ceil(len(horizons)/2))

        fig, axes = plt.subplots(rows, 2, figsize=(18, 4*rows), sharex=True) if rows > 1 else plt.subplots(len(horizons), 1, figsize=(18, 5*len(horizons)), sharex=True)
        axes = axes.flatten() if len(horizons) > 1 else [axes]

        # Create a list to store all line objects and their labels for the common legend
        lines = []
        labels = []

        for i, d in enumerate(horizons):
            ax = axes[i]
            ax.axhline(0, color="black", alpha=0.5, linewidth=1)

            # counter for color
            plotted_securities = 0

            for j, series in enumerate(self.series_list):
                returns = series.df
                label = series.label
                for k, col in enumerate([c for c in returns.columns if c.endswith(f"_{d}_return")]):

                    plot_label = f"{label} - {col.replace(f'_{d}_return', '')}" if "rh" in col else f"{col.replace(f'_{d}_return', '')}"
                
                    # find index 
                    color_index = plotted_securities % Plotter.index_counter
                    
                    line = sns.lineplot(x=returns.index, y=returns[col], 
                                    ax=ax, color=self.palette[color_index], linewidth=1.0,
                                    label='_nolegend_')  # Hide individual legends
                    
                    # Only add to legend list the first time we see this series (in the first subplot)
                    if i == 0:
                        lines.append(line.lines[-1])
                        labels.append(plot_label)

                    # increase counter for color
                    plotted_securities += 1

            ax.set_title(f"Horizon: {d} days")
            ax.tick_params(axis='x', rotation=0)
            ax.set_ylabel("Log Returns" if i % 2 == 0 else "")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.grid(True, linestyle='--', alpha=0.6)
            # Remove individual legends
            ax.get_legend().remove() if ax.get_legend() is not None else None

        if custom_labels is not None:
            assert len(labels) == len(custom_labels), f"Custom labels must be of same lenght as the series. Required length: {len(labels)}, current length; {len(custom_labels)}"
            labels = custom_labels

        # Create a single legend outside all subplots
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.05, 0.98),
                ncol=(len(labels)//4+1), frameon=True, fontsize='medium')

        fig.suptitle(title, fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted rect to leave space for the legend

        if save:
            out_dir = f"{self.images_dir}/{file_name}"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            plt.show()


    def plot_returns_kdes(self, 
                        custom_labels: List[str] = None, 
                        save:bool=False, 
                        file_name:str="kde_plot.png", 
                        title:str="Rolling Market vs Retail Returns Distribution Across Horizons", 
                        show:bool=True):
        
        if not self.series_list:
            raise ValueError("No return series provided.")

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Use horizons from the first series (assumed common)
        horizons = self.series_list[0].horizons
        cols = int(np.ceil(len(horizons)/2))
        
        if len(horizons) > 1:
            if cols > 1:
                fig, axes = plt.subplots(2, cols, figsize=(18, 10))
            else:
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        else:
                fig, axes = plt.subplots(1, 1, figsize=(18, 5))

            
        axes = axes.flatten() if len(horizons) > 1 else [axes]

        # Create a list to store all line objects and their labels for the common legend
        lines = []
        labels = []

        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]

            # counter for color
            plotted_securities = 0

            for j, series in enumerate(self.series_list):
                returns = series.df
                label = series.label
                for k, col in enumerate([c for c in returns.columns if c.endswith(f"_{d}_return")]):

                    plot_label = f"{label} - {col.replace(f'_{d}_return', '')}" if "rh" in col else f"{col.replace(f'_{d}_return', '')}"

                    # find index 
                    color_index = plotted_securities % Plotter.index_counter
                    
                    line = sns.kdeplot(data=returns[col], ax=ax, 
                                        color=self.palette[color_index], linewidth=1.0,
                                        label='_nolegend_')  # Hide individual legends
                    
                    # Only add to legend list the first time we see this series (in the first subplot)
                    if i == 0:
                        # Get the last line added to the current axes
                        lines.append(ax.get_lines()[-1])
                        labels.append(plot_label)

                    # increase counter for color
                    plotted_securities += 1
        
            # formatting
            ax.set_title(f"Horizon: {d} days")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
            ax.tick_params(axis='x', rotation=0)
            ax.set_ylabel("Density" if (i % cols) == 0 else "") # Divide by rows
            ax.set_xlabel("Log Returns")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        if custom_labels is not None:
            assert len(labels) == len(custom_labels), f"Custom labels must be of same lenght as the series. Required length: {len(labels)}, current length; {len(custom_labels)}"
            labels = custom_labels

        # Create a single legend with the specified positioning
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.05, 0.98),
                ncol=(len(labels)//4+1), frameon=True, fontsize='medium')

        # Set main figure title
        fig.suptitle(title, fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        if save:
            out_dir = f"{self.images_dir}/{file_name}"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            # Show plot
            plt.show()

    def plot_returns_cdfs(self, 
                        custom_labels: List[str] = None, 
                        save=False, 
                        file_name:str="cdf_plot.png", 
                        title:str="Empirical CDF of Returns Across Horizons", 
                        show:bool=True):
        if not self.series_list:
            raise ValueError("No return series provided.")
        
        # Apply Seaborn styling
        sns.set_style("whitegrid")
        
        # Use horizons from the first series (assumed common)
        horizons = self.series_list[0].horizons
        cols = int(np.ceil(len(horizons)/2))
        
        if cols > 1:
            fig, axes = plt.subplots(2, cols, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        axes = axes.flatten()        
        
        # Create a list to store all line objects and their labels for the common legend
        lines = []
        labels = []
        
        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]

            # counter for color
            plotted_securities = 0

            for j, series in enumerate(self.series_list):
                returns = series.df
                label = series.label
                for k, col in enumerate([c for c in returns.columns if c.endswith(f"_{d}_return")]):

                    plot_label = f"{label} - {col.replace(f'_{d}_return', '')}" if "rh" in col else f"{col.replace(f'_{d}_return', '')}"

                    # find index 
                    color_index = plotted_securities % Plotter.index_counter

                    # Plot CDF for each column with no legend label
                    line = self._plot_cdf(returns[col], '_nolegend_', ax, self.palette[color_index])
                    
                    # Only add to legend list the first time we see this series (in the first subplot)
                    if i == 0:
                        lines.append(line)
                        labels.append(plot_label)
            
                    # increase counter for color
                    plotted_securities += 1

                # Set subplot title
                ax.set_title(f"Horizon: {d}")
                        
            # formatting
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis='x', rotation=0)
            ax.set_xlabel("Returns")
            ax.set_ylabel("Probability" if (i % cols) == 0 else "")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        if custom_labels is not None:
            assert len(labels) == len(custom_labels), f"Custom labels must be of same lenght as the series. Required length: {len(labels)}, current length; {len(custom_labels)}"
            labels = custom_labels

        # Create a single legend with the specified positioning
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.05, 0.98),
                ncol=(len(labels)//4+1), frameon=True, fontsize='medium')
        
        # Set main figure title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            out_dir = f"{self.images_dir}/{file_name}"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")
        
        if show:
            # Show plot
            plt.show()

    def _plot_cdf(self, data, label, ax, color):
        """Helper method to plot a single CDF on the given axis"""
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        line = ax.plot(sorted_data, cdf, label=label, color=color, linewidth=1.0)[0]
        return line  # Return the Line2D object for legend
        

    def plot_ssd_comparison(self, 
                            series_a:pd.Series, 
                            series_b:pd.Series,
                            name_a:str=None,
                            name_b:str=None,
                            save:bool=False, 
                            file_name:str="SSD", 
                            title=None, 
                            show:bool=True):
        """
        Test for second-order stochastic dominance and visualize the results.
        
        Parameters:
        col_a : str
            Column name for the first return series
        col_b : str
            Column name for the second return series
        name_a : str, optional
            name for series_a
        name_b : str, optional
            name for series_b
        save : bool, optional
            Whether to save the plot
        name : str, optional
            Name of the saved file
        title : str, optional
            Plot title
        show : bool, optional
            Whether to show the plot
            
        Returns:
        dominance: bool
            True if series A dominates series B
        """
        # Initialize names
        name_a = name_a if name_a is not None else "series_a"
        name_b = name_b if name_b is not None else "series_b"

        # Ensure sns is used
        sns.set_style("whitegrid")

        # Run the SSD test
        dominance, int_cdf_a, int_cdf_b, x_grid, confidence = test_second_order_stochastic_dominance(series_a=series_a, series_b=series_b)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Original normalized CDFs
        returns_a = series_a.dropna().values
        returns_b = series_b.dropna().values

        # Build cdf        
        x_a = np.sort(returns_a)
        x_b = np.sort(returns_b)
        cdf_a = np.arange(1, len(x_a) + 1) / len(x_a)
        cdf_b = np.arange(1, len(x_b) + 1) / len(x_b)
        
        ax1.plot(x_a, cdf_a, label=f"{name_a} CDF", linewidth=1, color=self.palette[0])
        ax1.plot(x_b, cdf_b, label=f"{name_b} CDF", linewidth=1, color=self.palette[1])
        ax1.set_title("Normalized CDFs")
        ax1.set_xlabel("Normalized Returns")
        ax1.set_ylabel("Cumulative Probability")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Integrated CDFs
        ax2.plot(x_grid, int_cdf_a, label=f"{name_a} Integrated CDF", linewidth=1, color=self.palette[0])
        ax2.plot(x_grid, int_cdf_b, label=f"{name_b} Integrated CDF", linewidth=1, color=self.palette[1])
        ax2.set_title("Integrated CDFs")
        ax2.set_xlabel("Normalized Returns")
        ax2.set_ylabel("Integrated CDF Value")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add dominance message
        if dominance:
            dom_msg = f"{name_a} dominates {name_b} (SSD)"
        else:
            dom_msg = f"{name_a} does not dominate {name_b} (SSD), {confidence:.2f}% of points support dominance"
        
        if title:
            fig.suptitle(f"{title}\n{dom_msg}", fontsize=16)
        else:
            fig.suptitle(dom_msg, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save with the name
        if save:
            out_dir = f"{self.images_dir}/{file_name}"
            plt.savefig(out_dir, dpi=400, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            plt.show()
        
        return dominance