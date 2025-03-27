import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from src.DataLoader import DataLoader

class Analyzer():
    def __init__(self, dl_kwargs:dict={"handle_nans":"drop"}):
        # Instantiate Dataloader
        handle_nans = dl_kwargs.get("handle_nans", None)
        self.dl = DataLoader(handle_nans=handle_nans)
        
        # Memorize important dfs
        self.df_merged = self.dl.merge_dfs()
        self.df_sp = self.df_merged[self.df_merged["ticker"]=="VOO"]


    def _build_levels(self):
        # Build Portfolio using Popularity
        self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["prc_adj"]

        # Rename the SP500 portfolio
        df_sp = self.df_sp.copy()
        df_sp = df_sp[["date", "mc"]].set_index("date", drop=True)
        df_sp = df_sp.rename(columns={"mc":"voo"})

        # merge the levels
        levels = self.df_merged[["date", "rh_portfolio", "mc"]].groupby("date").sum()
        levels = levels.merge(df_sp, on="date")
        
        return levels        


    def build_returns(self, horizons:list=[5,15,30, 60, 120], start_date=None):
        if start_date != None:
            self.df_merged = self.dl.merge_dfs(start_date=start_date)

        # Retrieve levels
        levels = self._build_levels()

        if start_date != None:
            levels = levels[levels.index>=start_date]

        # Build returns df, applying logs ensure additivity of returns
        returns = np.log(levels / levels.shift(1)).fillna(0)
        returns.index = pd.to_datetime(returns.index)

        # Add the cumulative returns for the whole period and keep smaller values
        #horizons.append(len(returns))
        horizons = [h for h in horizons if h <= len(returns)]
        horizons = sorted(list(set(horizons)))

        for d in horizons: # delete min_periods if you want to start at date d and not before
            returns[f"mc_{d}_return"] = returns["mc"].rolling(d, min_periods=1).sum()
            returns[f"rh_portfolio_{d}_return"] = returns["rh_portfolio"].rolling(d, min_periods=1).sum()
            returns[f"voo_{d}_return"] = returns["voo"].rolling(d, min_periods=1).sum()

            # Corr 
            #returns[f"mc_rh_{d}_corr"] = returns["mc"].rolling(d, min_periods=1).corr(returns["rh_portfolio"])
            #returns[f"rh_voo_{d}_corr"] = returns["rh_portfolio"].rolling(d, min_periods=1).corr(returns["voo"])



        return returns, horizons

    def plot_returns_timeseries(self, returns_kwargs:dict={"horizons":[5,15,30, 60, 120], "start_date":None}):
        # Retrieve Returns 
        returns, horizons = self.build_returns(**returns_kwargs)

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True, sharey=True)
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
            sns.lineplot(x=returns.index, y=returns[f"mc_{d}_return"], label=f"Market returns", ax=ax, color=colors[0])
            # Plot Retail Market Cap returns
            sns.lineplot(x=returns.index, y=returns[f"rh_portfolio_{d}_return"], label=f"Retail returns", ax=ax, color=colors[1])
            # Plot VOO returns
            sns.lineplot(x=returns.index, y=returns[f"voo_{d}_return"], label=f"VOO returns", ax=ax, color=colors[2])


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

        # Show plot
        plt.show()

    def plot_kdes(self, returns_kwargs:dict={"horizons":[5,15,30, 60, 120], "start_date":None}):
        # Retrieve Returns 
        returns, horizons = self.build_returns(**returns_kwargs)

        # Apply Seaborn styling
        sns.set_style("whitegrid")

        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        # Define colors using Seaborn palette
        colors = sns.color_palette()


        # Iterate through horizons and create subplots
        for i, d in enumerate(horizons):
            ax = axes[i]

            sns.kdeplot(data=returns[f"mc_{d}_return"], label=f"Market Distribution",  ax=ax, color=colors[0])
            sns.kdeplot(data=returns[f"rh_portfolio_{d}_return"], label=f"RH Distribution", ax=ax, color=colors[1])
            sns.kdeplot(data=returns[f"voo_{d}_return"], label=f"VOO Distribution",  ax=ax, color=colors[2])

            # Set subplot title
            ax.set_title(f"Horizon: {d}")

            # Clean up x-axis labels
            if i > 2:  # Bottom row only
                ax.set_xlabel("Returns")
            else:
                ax.set_xlabel("")



            # Improve X-axis formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=0)


            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Set main figure title
        fig.suptitle("Rolling Market vs Retail Returns Distribution Across Horizons", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Show plot
        plt.show()

