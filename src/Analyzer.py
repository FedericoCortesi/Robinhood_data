import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from typing import Optional

from . import DataLoader
from .utils.metrics import log_ma_returns
from .utils.helpers import load_data_paths
from .utils.params import ReturnParams
from .utils.enums import WeightsMethod

# Setup logger
import logging
from .utils.custom_formatter import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

from . import DataLoader

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent

class Analyzer:
    def __init__(self,
                 weights_method: str | WeightsMethod = "stocks",
                 stocks_only: bool = False,
                 include_dividends: bool = False,
                 compare_tickers: Optional[list[str]] = None,
                 dl_kwargs: Optional[dict] = None,
                 return_params: Optional[ReturnParams] = None):
        """
        Initialize the Analyzer object for comparing performance metrics between tickers.
        
        Parameters
        ----------
        
        weights_methods : str {stocks, wealth}, default="stocks"
            String to build the rh portfolio weights based on wealth or stocks imputation. 
            - stocks: assumes each holding in the dataframe represents one stock
            - wealth: assumes `popularity` is a proxy for percentage of welath held in a certain stock

        stocks_only : bool, default=False
            Bool to build the merged dataframe taking into account only stocks (shrcd=10 or shrcd=11) or not. 

        include_dividemds : bool, default=False
            Bool to include the RH index with dividends or not. 

        compare_tickers : list, default=["VOO"]
            List of ticker symbols to compare against the Robinhood portfolio.
        
        dl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to the DataLoader.
            
        return_params : dict, default parameters include:
            Dictionary of parameters for calculating returns:
            - horizons : set, default={5,15,30,60,120}
                Time horizons (in days) for calculating returns.
            - start_date : str or None, default=None
                Start date for the analysis period. If None, uses all available data.
            - end_date : str or None, default=None
                End date for the analysis period. If None, uses all available data.
            - cumulative : bool, default=True
                If True, returns are calculated as cumulative over the horizon.
            - append_start : bool, default=True
                If True, includes the starting point in the return series.
        
        Returns
        -------
        None
        
        Notes
        -----
        This class merges dataframes containing price and popularity data and 
        provides methods for analyzing returns distribution and stochastic dominance.
        """     
        logger.info(f"{'#' * 30} Analysis Started {'#' * 30}")
        # Safe Enum conversion
        if isinstance(weights_method, str):
            try:
                weights_method = WeightsMethod(weights_method.lower())
            except ValueError:
                raise ValueError(f"`weights_method` must be one of {[m.value for m in WeightsMethod]}")

        self.weights_method = weights_method
        
        # Save attributes
        self.stocks_only = stocks_only
        
        # Instantiate Dataloader
        dl_kwargs = dl_kwargs if dl_kwargs is not None else {}
        self.dl = DataLoader(**dl_kwargs)
        
        # Memorize tickers to compare and return params
        self.compare_tickers = compare_tickers if compare_tickers is not None else ["VOO"]
        self.return_params = return_params if return_params is not None else ReturnParams()
 
        # Memorize important dfs
        # "mc" variable used to be here, decided to delete it as i don't care about the "market index" built on RH data.
        # Previously, the "market index"_t was just \sum_{i=1}^N P_{i,t}\cdot S_{i,t}
        self.df_merged = self.dl.merge_dfs(stocks_only=self.stocks_only)
        
        # Filter columns
        cols_to_keep = ["date", "prc_adj", "prc_adj_div", "popularity", "ticker", "ret"]
        
        # filter for dividends
        if not include_dividends:
            cols_to_keep = [col for col in cols_to_keep if "div" not in col]
        
        # Find those contained in the df
        cols_to_keep = [col for col in cols_to_keep if col in self.df_merged]
        self.df_merged = self.df_merged[cols_to_keep]

        # Define self.colors using Seaborn palette
        self.colors = sns.color_palette("muted")

        # Define images directory
        parent_dir = CURRENT_DIR.parents[0]
        self.data_paths = load_data_paths()

        self.df_robinhood_path = parent_dir / self.data_paths["df_robinhood_path"]
        self.images_dir = "../non_code/latex/images"

    def _extract_relevant_tickers(self):
        inner_compare = self.compare_tickers.copy()

        df_merged = self.df_merged.copy()

        # Extract only relevant tickers
        df_clean = df_merged[["prc_adj", "date", "ticker"]][df_merged["ticker"].isin(self.compare_tickers)]
        df_clean = df_clean

        # Initialize df_out
        df_out = None

        if self.compare_tickers:
            # Iterate over each ticker to create a dataframe 
            for i, col in enumerate(self.compare_tickers):

                if col not in df_clean["ticker"].unique():
                    message = "Empty dataframe produced for ticker: {col}." 
                    message = message + f" Maybe {col} is not a stock?" if self.stocks_only else message
                    logger.warning(message)

                    # Remove it from tickers to avoid problems when plotting 
                    message = f"Removing {col} from {inner_compare}"
                    logger.warning(message)
                    inner_compare.remove(col)
                    continue
                
                # Filter and rename
                df_etf = df_clean[df_clean["ticker"]==col]
                df_etf = df_etf.rename(columns={"prc_adj":col})
                df_etf = df_etf[[col, "date"]]

                if df_out is None:
                    df_out = df_etf

                # Merge in case you have more than one ticker
                else:
                    df_out = df_out.merge(df_etf, on="date")


        if df_out is None:
            logger.warning(f"Empty dataframe produced for tickers: {self.compare_tickers}")

            # Now set compare tickers to found tickers
            self.compare_tickers = inner_compare
            return
        
        # Set index
        df_out = df_out.set_index("date")

        return df_out
    
    

    def build_levels(self)->pd.DataFrame:
        """
        Build a dataframe with the daily price of `self.compare_tickers` and the robinhood portfolio.
        
        Parameters
        ----------
        None

        Returns
        -------
        levels : pd.DataFrame, a dataframe containing the daily value of the tickers and reference index (if `self.weights_method` is set to `stocks`)
        """
        
        if self.weights_method == WeightsMethod.STOCKS:
            # Build Portfolio using Popularity
            self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["prc_adj"]
            # Include dividends
            if "prc_adj_div" in self.df_merged.columns:
                self.df_merged["rh_portfolio_div"] = self.df_merged["popularity"] * self.df_merged["prc_adj_div"]

        elif self.weights_method == WeightsMethod.WEALTH:
            self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["ret"]
            self.df_merged["rh_portfolio"] = self.df_merged["rh_portfolio"].apply(lambda x: np.log(x+1))
        else:
            raise ValueError(f"Unsupported weights_method: {self.weights_method}")
        

        # Using the sum and including "mc" would build the "market index". Excluded as it's not very relevant 
        if "rh_portfolio_div" in self.df_merged.columns:
            levels = self.df_merged[["date", "rh_portfolio", "rh_portfolio_div"]].groupby("date").sum()
        else:
            levels = self.df_merged[["date", "rh_portfolio"]].groupby("date").sum()


        # Obtain the dataframe with relevant tickers
        df_tickers = self._extract_relevant_tickers()

        # Merge the levels
        if df_tickers is not None:
            levels = levels.merge(df_tickers, on="date")

        # set a flag for returns
        if self.weights_method == WeightsMethod.WEALTH:
            self.returns_columns = ["rh_portfolio"]
        else:
            self.returns_columns = []

        return levels        



    def build_returns(self):
        # Get params
        start_date = self.return_params.start_date
        end_date = self.return_params.end_date
        
        # Retrieve levels
        levels = self.build_levels()

        # Filter
        if start_date is not None:
            levels = levels[levels.index>=start_date]
        if end_date is not None:
            levels = levels[levels.index<=end_date]

        # call function with the params
        result = log_ma_returns(
            levels=levels, 
            return_params=self.return_params,
            returns_columns=self.returns_columns)
        
        return result
        

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