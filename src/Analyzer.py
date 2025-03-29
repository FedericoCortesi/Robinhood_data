import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import DataLoader
from .utils import log_ma_returns

class Analyzer():
    def __init__(self, compare_tickers:list=["VOO"], dl_kwargs:dict={"handle_nans":"drop"}):
        # Instantiate Dataloader
        handle_nans = dl_kwargs.get("handle_nans", None)
        self.dl = DataLoader(handle_nans=handle_nans)
        
        # Memorize tickers to compare
        self.compare_tickers = compare_tickers

        # Memorize important dfs
        self.df_merged = self.dl.merge_dfs()
        self.df_sp = self.df_merged[self.df_merged["ticker"]=="VOO"]
      


    def _extract_relevant_tickers(self):
        df_merged = self.df_merged.copy()

        # Extract only relevant tickers
        df_clean = df_merged[df_merged["ticker"].isin(self.compare_tickers)]
        df_clean = df_clean[["prc_adj", "date", "ticker"]]

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

        # merge the levels
        levels = self.df_merged[["date", "rh_portfolio", "mc"]].groupby("date").sum()
        levels = levels.merge(df_tickers, on="date")
        
        return levels        


    def build_returns(self, horizons:list=[5,15,30, 60, 120], start_date=None):
        if start_date != None:
            self.df_merged = self.dl.merge_dfs(start_date=start_date)

        # Retrieve levels
        levels = self._build_levels()

        if start_date != None:
            levels = levels[levels.index>=start_date]

        result = log_ma_returns(levels=levels, horizons=horizons)
        
        return result
        


    def plot_returns_timeseries(self, returns_kwargs:dict={"horizons":[5,15,30, 60, 120], "start_date":None}, save:bool=False):
        # Retrieve Returns 
        returns, horizons = self.build_returns(**returns_kwargs)

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
        axes = axes.flatten()

        # Define colors using Seaborn palette
        colors = sns.color_palette()

        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]
            
            # Draw horizontal line at 0
            ax.axhline(0, color="black", alpha=0.5, linewidth=1)
            if d < len(returns):
                ax.axvline(returns.index[d], color="black", alpha=0.5, linewidth=1)

            # Plot Market Cap returns
            sns.lineplot(x=returns.index, y=returns[f"mc_{d}_return"], label=f"Market returns", ax=ax, color=colors[0], markers=True, markersize=5, linewidth=1.0)
            # Plot Retail Market Cap returns
            sns.lineplot(x=returns.index, y=returns[f"rh_portfolio_{d}_return"], label=f"Retail returns", ax=ax, color=colors[1], markers=True, markersize=5, linewidth=1.0)
            # Plot ticker returns
            for i, ticker in enumerate(self.compare_tickers):
                sns.lineplot(x=returns.index, y=returns[f"{ticker}_{d}_return"], label=f"{ticker} returns", ax=ax, color=colors[2+i], markers=True, markersize=5, linewidth=1.0)


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
        fig.suptitle("Rolling Market vs Retail Returns Across Horizons", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            out_dir = "../images/returns_plot.png"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")


        # Show plot
        plt.show()

    def plot_returns_kdes(self, returns_kwargs:dict={"horizons":[5,15,30, 60, 120], "start_date":None}, save=False):
        # Retrieve Returns 
        returns, horizons = self.build_returns(**returns_kwargs)

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Define colors using Seaborn palette
        colors = sns.color_palette()


        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]

            sns.kdeplot(data=returns[f"mc_{d}_return"], label=f"Market Distribution",  ax=ax, color=colors[0], linewidth=1.0)
            sns.kdeplot(data=returns[f"rh_portfolio_{d}_return"], label=f"RH Distribution", ax=ax, color=colors[1], linewidth=1.0)
            for i, ticker in enumerate(self.compare_tickers):
                sns.kdeplot(data=returns[f"{ticker}_{d}_return"], label=f"{ticker} Distribution",  ax=ax, color=colors[2+i], linewidth=1.0)

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
        fig.suptitle("Rolling Market vs Retail Returns Distribution Across Horizons", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            out_dir = "../images/distributions_plot.png"
            plt.savefig(out_dir, dpi=600, bbox_inches='tight')
            print(f"file saved at {out_dir}")

        # Show plot
        plt.show()
 