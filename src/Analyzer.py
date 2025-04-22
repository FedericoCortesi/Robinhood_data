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
from .utils.enums import WeightsApplication

# Setup logger
import logging
from .utils.custom_formatter import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

from . import DataLoader

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent

class Analyzer:
    def __init__(self,
                 weights_application: str | WeightsApplication = "number",
                 stocks_only: bool = False,
                 include_dividends: bool = False,
                 compare_tickers: Optional[list[str]] = None,
                 dl_kwargs: Optional[dict] = None,
                 return_params: Optional[ReturnParams] = None):
        """
        Initialize the Analyzer object for comparing performance metrics between tickers.
        
        Parameters
        ----------
        
        weights_applications : str {number, wealth}, default="number"
            String to build the rh portfolio weights based on wealth or number imputation. 
            - number: assumes each holding in the dataframe represents one stock
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
        if isinstance(weights_application, str):
            try:
                weights_application = WeightsApplication(weights_application.lower())
            except ValueError:
                raise ValueError(f"`weights_application` must be one of {[m.value for m in WeightsApplication]}")

        self.weights_application = weights_application
        
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
        cols_to_keep = ["date", "prc_adj", "prc_adj_div", "divamt", "popularity", "ticker", "ret", "shrcd"]
        
        # filter for dividends
        self.include_dividends = include_dividends
        if not self.include_dividends:
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
                    message = f"Empty dataframe produced for ticker: {col}." 
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
    

    def _reinvest_dividends_value(self,value_array:np.ndarray) -> np.ndarray:
        """
        Build an array of dividends accounting for their growth if they were inveested with the same returns of the `value_array`.
        
        Parameters
        ----------
        value_array : np.ndarray, array of values of the portfolio 

        Returns
        -------
        total_reinvested : np.ndarray, array of dividends accounting for daily returns 
                
        """
        # compute gross returns form value_array, it represents the returns of the portfolio
        ret = value_array / value_array.shift(1)
        ret = ret.fillna(1) # because at t=1 the return is 1

        # get the returns from the merged df
        div_array = self.df_merged[["divamt", "date", "popularity"]]
        div_array["divamt"] = div_array["divamt"] * div_array["popularity"] # Normalize by weight
        div_array = div_array[["divamt", "date"]].groupby("date").sum() 
        div_array = div_array.values.flatten() # flatten becuase it is a list of lists

        # Initialize the array to store results
        total_reinvested = []

        # iterate over each date
        for i, val in enumerate(div_array):

            # get the value of the prior date
            prior_value = total_reinvested[-1] if i != 0 else 0

            # add the cash inflow 
            value = prior_value + val

            # multiply by the daily returns to see how much is the value after
            value = value * ret[i]

            # append to list
            total_reinvested.append(float(value))

        total_reinvested = np.array(total_reinvested)
        
        return total_reinvested

    

    def build_levels(self)->pd.DataFrame:
        """
        Build a dataframe with the daily price of `self.compare_tickers` and the robinhood portfolio.
        
        Parameters
        ----------
        None

        Returns
        -------
        levels : pd.DataFrame, a dataframe containing the daily value of the tickers and reference index (if `self.weights_application` is set to `stocks`)
        """
        
        if self.weights_application == WeightsApplication.NUMBER:
            # Build Portfolio using Popularity
            self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["prc_adj"]

        elif self.weights_application == WeightsApplication.WEALTH:
            self.df_merged["rh_portfolio"] = self.df_merged["popularity"] * self.df_merged["ret"]
        else:
            raise ValueError(f"Unsupported weights_application: {self.weights_application}")
        

        # Using the sum and including "mc" would build the "market index". Excluded as it's not very relevant 
        if self.include_dividends:
            levels = self.df_merged[["date", "rh_portfolio"]].groupby("date").sum()
            reinvested_dividends = self._reinvest_dividends_value(levels["rh_portfolio"])
            levels["rh_portfolio_div"] = reinvested_dividends + levels["rh_portfolio"]

        else:
            levels = self.df_merged[["date", "rh_portfolio"]].groupby("date").sum()
            
        # Convert to log returns
        if self.weights_application == WeightsApplication.WEALTH: 
            levels["rh_portfolio"] = levels["rh_portfolio"].apply(lambda x: np.log(x+1))


        # Obtain the dataframe with relevant tickers
        df_tickers = self._extract_relevant_tickers()

        # Merge the levels
        if df_tickers is not None:
            levels = levels.merge(df_tickers, on="date")

        # set a flag for returns
        if self.weights_application == WeightsApplication.WEALTH:
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
        

    

