import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

from . import DataLoader
from .utils import log_ma_returns, setup_custom_logger

# Setup logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

class Analyzer():
    def __init__(self, compare_tickers:list=["VOO"], 
                 dl_kwargs:dict={"handle_nans":"drop"}, 
                 return_params:dict={"horizons":{5,15,30, 60, 120}, # Set so it doesn't go on indefinitely
                                     "start_date":None, 
                                     "end_date":None, 
                                     "cumulative":True,
                                     "append_start":True}):
        
        # Instantiate Dataloader
        handle_nans = dl_kwargs.get("handle_nans", None)
        self.dl = DataLoader(handle_nans=handle_nans)
        
        # Memorize tickers to compare and return params
        self.compare_tickers = compare_tickers
        self.return_params = return_params
 
        # Memorize important dfs
        # "mc" variable used to be here, decided to delete it as i don't care about the "market index" built on RH data.
        # Previously, the "market index"_t was just \sum_{i=1}^N P_{i,t}\cdot S_{i,t}
        self.df_merged = self.dl.merge_dfs(columns=["date", "prc_adj", "popularity", "ticker"])

        # Define self.colors using Seaborn palette
        self.colors = sns.color_palette("muted")

        # Define images directory
        self.images_dir = "../non_code/latex/images"

        logger.info("Analyzer instantiated!")




    def _extract_relevant_tickers(self):
        df_merged = self.df_merged.copy()

        # Extract only relevant tickers
        df_clean = df_merged[["prc_adj", "date", "ticker"]][df_merged["ticker"].isin(self.compare_tickers)]
        df_clean = df_clean

        # Iterate over each ticker to create a dataframe 
        for i, col in enumerate(self.compare_tickers):
            # Filter and rename
            df_etf = df_clean[df_clean["ticker"]==col]
            df_etf = df_etf.rename(columns={"prc_adj":col})
            df_etf = df_etf[[col, "date"]]

            if i<1:
                df_out = df_etf

            # Merge in case you have more than one ticker
            else:
                df_out = df_out.merge(df_etf, on="date")

        # Set index
        df_out = df_out.set_index("date")

        return df_out

    def _build_levels(self):
        # Build Portfolio using Popularity
        self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["prc_adj"]

        # Obtain the dataframe with relevant tickers
        df_tickers = self._extract_relevant_tickers()

        # Merge the levels
        # Using the sun and including "mc" would build the "market index". Excluded as it's not very relevant 
        levels = self.df_merged[["date", "rh_portfolio"]].groupby("date").sum()
        levels = levels.merge(df_tickers, on="date")
        
        return levels        


    def build_returns(self):
        # Get params
        start_date = self.return_params.get("start_date")
        end_date = self.return_params.get("end_date")
        horizons = set(self.return_params.get("horizons")) # Ensure consistency and remove double entries
        cumulative = self.return_params.get("cumulative")
        append_start = self.return_params.get("append_start")
        
        # Retrieve levels
        levels = self._build_levels()

        # Filter
        if start_date != None:
            levels = levels[levels.index>=start_date]
        if end_date != None:
            levels = levels[levels.index<=end_date]

        result = log_ma_returns(levels=levels, horizons=horizons, cumulative=cumulative, append_start=append_start)
        
        return result
        


    def plot_returns_timeseries(self, 
                                save:bool=False, 
                                name:str="returns_plot.png", 
                                title:str="Rolling Market vs Retail Returns Across Horizons", 
                                show:bool=True):
        # Retrieve Returns 
        returns, horizons = self.build_returns()

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with subplots, determine dinamycally the number
        rows = int(np.ceil(len(horizons)/2))
        
        if rows > 1:
            fig, axes = plt.subplots(rows, 2, figsize=(18, 12), sharex=True)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        axes = axes.flatten()

        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]
            
            # Draw horizontal line at 0
            ax.axhline(0, color="black", alpha=0.5, linewidth=1)
            if d < len(returns):
                ax.axvline(returns.index[d], color="black", alpha=0.5, linewidth=1)

            # Plot Market Cap returns
            #sns.lineplot(x=returns.index, y=returns[f"mc_{d}_return"], label=f"Market returns", ax=ax, color=self.colors[0], markers=True, markersize=5, linewidth=1.0)
            # Plot Retail Market Cap returns
            sns.lineplot(x=returns.index, y=returns[f"rh_portfolio_{d}_return"], label=f"RH returns", ax=ax, color=self.colors[1], markers=True, markersize=5, linewidth=1.0)
            # Plot ticker returns
            for i, ticker in enumerate(self.compare_tickers):
                sns.lineplot(x=returns.index, y=returns[f"{ticker}_{d}_return"], label=f"{ticker} returns", ax=ax, color=self.colors[2+i], markers=True, markersize=5, linewidth=1.0)


            # Set subplot title
            ax.set_title(f"Horizon: {d} days")

            # Improve X-axis formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=0)

            # Clean up y-axis labels
            if i % 2 == 0:  # Left column only
                ax.set_ylabel("Log Returns")
            else:
                ax.set_ylabel("")

            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc="upper left")

        # Set main figure title
        fig.suptitle(title, fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            out_dir = f"{self.images_dir}/{name}"
            plt.savefig(out_dir, dpi=400, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            # Show plot
            plt.show()

    def plot_returns_kdes(self, save=False, name:str="kde_plot.png", title:str="Rolling Market vs Retail Returns Distribution Across Horizons", show:bool=True):
        # Retrieve Returns 
        returns, horizons = self.build_returns()

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with automatic subplots
        cols = int(np.ceil(len(horizons)/2))
        
        if cols > 1:
            fig, axes = plt.subplots(2, cols, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes = axes.flatten()

        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]

            #sns.kdeplot(data=returns[f"mc_{d}_return"], label=f"Market Distribution",  ax=ax, color=self.colors[0], linewidth=1.0)
            sns.kdeplot(data=returns[f"rh_portfolio_{d}_return"], label=f"RH Distribution", ax=ax, color=self.colors[1], linewidth=1.0)
            for i, ticker in enumerate(self.compare_tickers):
                sns.kdeplot(data=returns[f"{ticker}_{d}_return"], label=f"{ticker} Distribution",  ax=ax, color=self.colors[2+i], linewidth=1.0)

            # Set subplot title
            ax.set_title(f"Horizon: {d}")

            # Clean up x-axis labels
            if i > 2:  # Bottom row only
                ax.set_xlabel("Returns")
            else:
                ax.set_xlabel("")

            # Improve X-axis formatting
            ax.tick_params(axis='x', rotation=0)


            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Set main figure title
        fig.suptitle(title, fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            out_dir = f"{self.images_dir}/{name}"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            # Show plot
            plt.show()

    def plot_returns_cdfs(self, save=False, name:str="cdf_plot.png", title:str="Empirical CDF of Returns Across Horizons", show:bool=True):
        # Retrieve Returns 
        returns, horizons = self.build_returns()
        
        # Apply Seaborn styling
        sns.set_style("whitegrid")
        
        # Create figure with automatic subplots
        cols = int(np.ceil(len(horizons)/2))
        
        if cols > 1:
            fig, axes = plt.subplots(2, cols, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        axes = axes.flatten()
        
        
        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]
            
            # Plot CDF for each column
            #self._plot_cdf(returns[f"mc_{d}_return"], "Market CDF", ax, self.colors[0])
            self._plot_cdf(returns[f"rh_portfolio_{d}_return"], "RH CDF", ax, self.colors[1])
            
            for j, ticker in enumerate(self.compare_tickers):
                self._plot_cdf(returns[f"{ticker}_{d}_return"], f"{ticker} CDF", ax, self.colors[2+j])
            
            # Set subplot title
            ax.set_title(f"Horizon: {d}")
            
            # Clean up x-axis labels
            if i > 2:  # Bottom row only
                ax.set_xlabel("Returns")
            else:
                ax.set_xlabel("")
            
            # Improve X-axis formatting
            ax.tick_params(axis='x', rotation=0)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
        
        # Set main figure title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            out_dir = f"{self.images_dir}/{name}"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")
        
        if show:
            # Show plot
            plt.show()

    def _plot_cdf(self, data, label, ax, color):
        """Helper method to plot a single CDF on the given axis"""
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, color=color, linewidth=1.0)


    def test_second_order_stochastic_dominance(self, col_a:str, col_b:str, df:pd.DataFrame=None):
        """
        Test for second-order stochastic dominance between two return series from a DataFrame.
        
        Parameters:
        - col_a : str
            Column name for the first return series
        - col_b : str
            Column name for the second return series
        - df : pandas.DataFrame, optional
            DataFrame containing the return series, calls `build_returns()` if omitted
            
        Returns:
        - tuple:
            - dominance: bool, True if series A dominates series B
            - integrated_cdf_a: numpy.array, integrated CDF values for series A
            - integrated_cdf_b: numpy.array, integrated CDF values for series B
            - x_grid: numpy.array, common x-axis values for the integrated CDFs
            - dominance_confidence: float, percentage of points where the dominance relation holds
        """    
        # Obtain return df if no df is provided
        if df is None:
            df = self.build_returns()[0]


        # Extract and drop NaN values
        returns_a = df[col_a].dropna().values
        returns_b = df[col_b].dropna().values
                
        # Create sorted arrays and CDFs
        x_a = np.sort(returns_a)
        x_b = np.sort(returns_b)

        # Go over each item in the array and get the cumulative probability
        cdf_a = np.arange(1, len(x_a) + 1) / len(x_a)
        cdf_b = np.arange(1, len(x_b) + 1) / len(x_b)
        
        # Create integrated CDFs (need to use common x-grid)
        x_grid = np.unique(np.concatenate([x_a, x_b]))
        x_grid.sort()  # Ensure the grid is sorted
        
        # Interpolate CDFs onto common grid
        cdf_a_interp = np.interp(x_grid, x_a, cdf_a, left=0)
        cdf_b_interp = np.interp(x_grid, x_b, cdf_b, left=0)
        
        # Calculate integrated CDFs
        dx = np.diff(x_grid, prepend=x_grid[0] - (x_grid[1] - x_grid[0]))
        integrated_cdf_a = np.cumsum(cdf_a_interp * dx)
        integrated_cdf_b = np.cumsum(cdf_b_interp * dx)
        
        # Test for SSD: A dominates B if integrated_cdf_a <= integrated_cdf_b for all points
        dominance_points = integrated_cdf_a <= integrated_cdf_b
        dominance = np.all(dominance_points)
        dominance_confidence = np.mean(dominance_points) * 100
        
        return dominance, integrated_cdf_a, integrated_cdf_b, x_grid, dominance_confidence

    

    def plot_ssd_comparison(self, col_a, col_b, df:pd.DataFrame=None, save:bool=False, name:str="SSD", title=None, show:bool=True):
        """
        Test for second-order stochastic dominance and visualize the results.
        
        Parameters:
        col_a : str
            Column name for the first return series
        col_b : str
            Column name for the second return series
        df : pandas.DataFrame, optional
            DataFrame containing the return series, calls `build_returns()` if omitted
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
        # Ensure sns is used
        sns.set_style("whitegrid")

        # Obtain return df if no df is provided
        if df is None:
            df = self.build_returns()[0]

        # Run the SSD test
        dominance, int_cdf_a, int_cdf_b, x_grid, confidence = self.test_second_order_stochastic_dominance(col_a=col_a, col_b=col_b, df=df)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Original normalized CDFs
        returns_a = df[col_a].dropna().values
        returns_b = df[col_b].dropna().values

        # Build cdf        
        x_a = np.sort(returns_a)
        x_b = np.sort(returns_b)
        cdf_a = np.arange(1, len(x_a) + 1) / len(x_a)
        cdf_b = np.arange(1, len(x_b) + 1) / len(x_b)
        
        ax1.plot(x_a, cdf_a, label=f"{col_a} CDF", linewidth=1, color=self.colors[0])
        ax1.plot(x_b, cdf_b, label=f"{col_b} CDF", linewidth=1, color=self.colors[1])
        ax1.set_title("Normalized CDFs")
        ax1.set_xlabel("Normalized Returns")
        ax1.set_ylabel("Cumulative Probability")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Integrated CDFs
        ax2.plot(x_grid, int_cdf_a, label=f"{col_a} Integrated CDF", linewidth=1, color=self.colors[0])
        ax2.plot(x_grid, int_cdf_b, label=f"{col_b} Integrated CDF", linewidth=1, color=self.colors[1])
        ax2.set_title("Integrated CDFs")
        ax2.set_xlabel("Normalized Returns")
        ax2.set_ylabel("Integrated CDF Value")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add dominance message
        if dominance:
            dom_msg = f"{col_a} dominates {col_b} (SSD)"
        else:
            dom_msg = f"{col_a} does not dominate {col_b} (SSD), {confidence:.1f}% of points support dominance"
        
        if title:
            fig.suptitle(f"{title}\n{dom_msg}", fontsize=14)
        else:
            fig.suptitle(dom_msg, fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save with the name
        if save:
            out_dir = f"{self.images_dir}/{name}"
            plt.savefig(out_dir, dpi=400, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        if show:
            plt.show()
        
        return dominance