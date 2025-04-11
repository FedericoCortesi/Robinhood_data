import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import logging
from .utils.custom_formatter import setup_custom_logger
# Setup logger
logger = setup_custom_logger(__name__, level=logging.INFO)

from .utils.helpers import load_data_paths
from .utils.enums import NaNHandling

from config import PROJECT_ROOT

class DataLoader:
    """Loads and preprocesses financial data from specified paths."""

    def __init__(self, 
                 handle_nans: str | NaNHandling = "drop", 
                 load_merged: bool = True, 
                 load_other_dfs: bool = False):
        """
        Initializes the DataLoader with data paths and processing options.

        Args:
            handle_nans (str): Strategy for handling NaNs ('fill', 'drop', 'keep').
            load_merged (bool): Whether to load the merged DataFrame.
            load_other_dfs (bool): Whether to load other DataFrames.
        """
            # Safe Enum conversion
        if isinstance(handle_nans, str):
            try:
                handle_nans = NaNHandling(handle_nans.lower())
            except ValueError:
                raise ValueError(f"`handle_nans` must be one of {[n.value for n in handle_nans]}")

        self.handle_nans = handle_nans
        self.load_merged = load_merged
        self.load_other_dfs = load_other_dfs

        self.data_paths = load_data_paths()
        self._set_file_paths()

        # Load the other dfs only in case its specificied (saves time in case you are just reading the csv)
        if load_other_dfs:
            self._load_robinhood_data()
            self._load_crsp_data()

    # Retrieves data paths from data_paths.json
    def _set_file_paths(self):
        """Sets the absolute file paths for the DataFrames."""

        self.df_robinhood_path = PROJECT_ROOT / self.data_paths["df_robinhood_path"]
        self.df_wrds_path = PROJECT_ROOT / self.data_paths["df_wrds_path"]
        self.df_crsp_path = PROJECT_ROOT / self.data_paths["df_crsp_path"]
        self.df_merged_path = PROJECT_ROOT / self.data_paths["df_merged_path"].format(self.handle_nans.value)

        logger.debug(f"self.df_robinhood_path: {self.df_robinhood_path}")
        logger.debug(f"self.df_wrds_path: {self.df_wrds_path}")
        logger.debug(f"self.df_crsp_path: {self.df_crsp_path}")
        logger.debug(f"self.df_merged_path: {self.df_merged_path}")

        # Check if the directory exists
        if not self.df_merged_path.exists():
            # if it doesnt exist force building the dataframe from scractch
            self.load_merged = False    


    def _load_robinhood_data(self):
        """"
        Loads the dataframe of all available robinhood securities.
        Nans are filled with zero and then deleted because some stocks have zeros instead of nans in the original data.
        """
        
        # Load csv
        df_rh = pd.read_parquet(self.df_robinhood_path)

        # Handle nans
        #df_rh = df_rh.fillna(0)
        #df_rh = df_rh.where(df_rh!=0, np.nan)


        # define row-wise sum
        df_rh["sum"] = df_rh.sum(axis=1)

        # Remove empty days
        df_rh = df_rh[df_rh["sum"]!=0]

        # rename columns 
        new_cols = []
        for row in df_rh.columns:
            n_col = row.replace(".","-")
            new_cols.append(n_col)
        df_rh.columns = new_cols

        # Count the number of non-nan stocks for each days
        df_rh.loc[:,"num_stocks"] = len(df_rh.columns) - df_rh.isna().sum(axis=1)

        # Handle nans
        if self.handle_nans == "fill":
            df_rh = df_rh.bfill(axis=0)
            df_rh = df_rh.ffill(axis=0)
        elif self.handle_nans == "drop":
            df_rh = df_rh.dropna(axis=1)
        else:
            pass

        # Save as attribute
        self.df_rh = df_rh
        
        logger.info("Robinhood data loaded")
        
        return
    
    def _load_crsp_data(self):
        """
        Loads CRSP Data
        """

        # Load csv
        df_crsp = pd.read_parquet(self.df_crsp_path)

        # Save
        #cols_to_drop = ["cfacshr_adj", "cfacpr_adj", "cfacshr", "cfacpr", "sprtrn"]
        #cols_to_drop = []
        #cols_to_drop = [col for col in df_crsp.columns if col in cols_to_drop]
        #df_crsp = df_crsp.drop(columns=cols_to_drop)
        self.df_crsp = df_crsp
        
        logger.info("CRSP data loaded")
        return
    
    
    def _build_df_crsp(self):
        """
        Loads the unedited WRDS file from memory and cleans it.

        Returns
        -----
        df_crsp : pd.DataFrame, the cleaned dataframe
        """
        # Load csv
        df_crsp = pd.read_parquet(self.df_wrds_path)

        # Build additional features
        # Adjust cfacshr to be constant in the period, it's a cumulative measure
        last_cf_dict = df_crsp.groupby("ticker").last()["cfacshr"].to_dict()
        df_crsp.loc[:,"cfacshr_last"] = df_crsp["ticker"].map(last_cf_dict)
        df_crsp.loc[:,"cfacshr_adj"] = df_crsp["cfacshr"] / df_crsp["cfacshr_last"] 

        # Adjust cfacpr to be constant in the period
        last_cf_dict = df_crsp.groupby("ticker").last()["cfacpr"].to_dict()
        df_crsp.loc[:, "cfacpr_last"] = df_crsp["ticker"].map(last_cf_dict)
        df_crsp.loc[:, "cfacpr_adj"] = df_crsp["cfacpr"] / df_crsp["cfacpr_last"] 

        # Adjust for negative prices, happens when the volume is 0
        df_crsp.loc[:, "prc"] = df_crsp["prc"].abs()

        # Adjust price and shares outstanding 
        df_crsp.loc[:,"prc_adj"] = df_crsp["prc"]/df_crsp["cfacpr_adj"] 
        df_crsp.loc[:,"shrout_adj"] = df_crsp["shrout"]*1000*df_crsp["cfacshr_adj"] 

        if "divamt" in df_crsp.columns:
            logger.debug("Handling dividends")
            df_crsp["divamt"] = df_crsp["divamt"].fillna(0) # Handle nans
            
            # Get the cumulative dividend amount for each ticker
            df_crsp["div_cum"] = df_crsp[["ticker", "divamt"]].groupby("ticker").cumsum()

            # Get the price with dividends ()
            df_crsp["prc_adj_div"] = df_crsp["prc_adj"] + df_crsp["div_cum"]


        # Prune instances that don't have shares outstanding
        df_crsp = df_crsp[df_crsp["shrout_adj"]!=0]

        # clean columns
        cols_to_drop = ["cfacshr", "cfacshr_last", "cfacpr", "cfacpr_last"]
        df_crsp = df_crsp.drop(columns=cols_to_drop)

        return df_crsp
    

    def merge_dfs(self, columns:list=None, stocks_only:bool=False):
        """
        Merges the CRSP and RH dataframe, either by loading from the data directory or by building it.
        """
        logger.debug(f"load_merged:  {self.load_merged}")
        if self.load_merged: # Access just read the file
            if columns:
                df = pd.read_parquet(self.df_merged_path, columns=columns)
            else:
                df = pd.read_parquet(self.df_merged_path)
        else:
            if not self.load_other_dfs: # Build the necessary files if not built in __init__
                self._load_robinhood_data()
                self._load_crsp_data()
            
            # Call the internal method to build the merged dataframe
            df = self._build_merged_df_from_crsp_rh()

        # If stock_only is true filter by the corresponding code before returning
        if stocks_only:
            # filter the codes correspondign to stocks
            df = df[(df["shrcd"]==11)|(df["shrcd"]==11)]

            # Compute popularity again
            df["popularity"] = df["holders"] / df[["date", "holders"]].groupby("date")["holders"].transform("sum")

            df["ticker"] = df["ticker"].cat.remove_unused_categories() # otherwise it keeps the old values for ticker in memory and is a problem for grouping etc
            df = df.reset_index(drop=True) # Ensure consistent index

        # Filter columns
        columns = columns if columns else df.columns 
        df = df[columns]

        return df

    def _compute_gross_returns(self, df:pd.DataFrame) -> pd.Series:
        """
        Given a pandas dataframe with tickers it groups by tickers and returns the gross returns by ticker.
        """
        # Get the first values for each ticker and paste them for every date
        firsts = (df[["ticker", "prc_adj"]].groupby('ticker').transform('first'))
        
        # Divide the column by the first value to compute gross returns
        result = df["prc_adj"] / firsts["prc_adj"]

        return result



    def _build_merged_df_from_crsp_rh(self, users:bool=False, start_date:str=None, end_date:str=None):
        # Filter dataframes
        self._filter_dfs_common_tickers()

        #--Change robinhood df to long form to match crsp's structure--#
        # Transpose
        df_rh_t = self.df_rh.copy()

        # Reset index so 'date' becomes a column
        self.df_rh_long = df_rh_t.reset_index().melt(id_vars='timestamp', var_name='ticker', value_name='holders')

        # Rename and fix features
        self.df_rh_long.rename(columns={'timestamp':'date'}, inplace=True)
        self.df_rh_long['date'] = pd.to_datetime(self.df_rh_long['date'])
        self.df_rh_long = self.df_rh_long.sort_values(by=["ticker", "date"])
        self.df_rh_long = self.df_rh_long.reset_index(drop=True)

        # Append users if true
        if users:
            self._concat_users_to_robinhood()
            self.df_rh_long["holders_adj"] = self.df_rh_long["holders"]/self.df_rh_long["users"]
        else:
            pass
        
        # Ensure 'date' columns are in datetime format in both dataframes
        self.df_rh_long['date'] = pd.to_datetime(self.df_rh_long['date'])
        self.df_crsp['date'] = pd.to_datetime(self.df_crsp['date'])

        # Filter if start/end date are passed
        if start_date:
            self.df_rh_long = self.df_rh_long[self.df_rh_long["date"]>=start_date]
            self.df_crsp = self.df_crsp[self.df_crsp["date"]>=start_date]

        if end_date:
            self.df_rh_long = self.df_rh_long[self.df_rh_long["date"]<=end_date]
            self.df_crsp = self.df_crsp[self.df_crsp["date"]<=end_date]

        # Merge both dataframes on 'date' and 'ticker'
        df_merged = self.df_rh_long.merge(self.df_crsp, on=['date', 'ticker'], how='inner')

        # Get the total number of trading days and filter out tickers with different values (missing or repeating data)
        trading_days = df_merged["date"].nunique()
        tickers_to_keep = df_merged["ticker"].value_counts()[df_merged["ticker"].value_counts()==trading_days].index
        df_merged = df_merged[df_merged["ticker"].isin(tickers_to_keep)]

        # Drop columns
        df_merged = df_merged.drop(columns=["permno"])

        #-- Build additional features --#

        # Return measures
        if "ret" in df_merged.columns: # Perform this operation only if the column is present in the downloaded df
            df_merged["log_returns"] = df_merged["ret"].apply(lambda x: np.log(x+1))
        else:
            #df_merged["daily_returns"] = df_merged.groupby("ticker")["prc_adj"].apply(lambda x: np.log(x / x.shift(1))).reset_index(level=0, drop=True).fillna(0)
            #df_merged["cumulative_returns"] = df_merged.groupby("ticker")["daily_returns"].cumsum()
            pass
        
        # Market cap
        df_merged['mc'] = df_merged['prc_adj'] * df_merged['shrout_adj']
        #df_merged['mc_retail'] = df_merged['prc_adj'] * df_merged["holders"]
        df_merged['market_weight'] = df_merged['mc'] / df_merged[["date", "mc"]].groupby("date")["mc"].transform("sum")
        #df_merged['retail_weight'] = df_merged['mc_retail'] / df_merged[["date", "mc_retail"]].groupby("date")["mc_retail"].transform("sum")
        
        # Measures with holders
        #df_merged["retail_ownership"] = df_merged["holders"] / df_merged["shrout_adj"]
        #df_merged["holders_change_pct"] = df_merged.groupby("ticker")["holders"].pct_change()
        df_merged["holders_change_pct"] = df_merged.groupby("ticker")["holders"].apply(lambda x: np.log(x / x.shift(1))).reset_index(level=0, drop=True)
        df_merged["holders_change_diff"] = df_merged.groupby("ticker")["holders"].diff()
        df_merged["total_holders"] = df_merged[["date", "holders"]].groupby("date").transform("sum")
        df_merged["popularity"] = df_merged["holders"] / df_merged[["date", "holders"]].groupby("date")["holders"].transform("sum")
        
        # Create total holders change
        total_holders_series = df_merged.drop_duplicates("date")[["date", "total_holders"]].set_index("date").sort_index()
        total_holders_change_pct = total_holders_series.pct_change().rename(columns={"total_holders": "total_holders_change_pct"})
        total_holders_change_diff = total_holders_series.diff().rename(columns={"total_holders": "total_holders_change_diff"})
        # Merge back to df_merged
        df_merged = df_merged.merge(total_holders_change_pct, on="date", how="left")
        df_merged = df_merged.merge(total_holders_change_diff, on="date", how="left")

        # Make ticker a category for faster parsing
        df_merged["ticker"] = df_merged["ticker"].astype("category")

        # Sort
        df_merged = df_merged.sort_values(by=["ticker", "date"])
        df_merged = df_merged.reset_index(drop=True)

        logger.info("DataFrames merged")
        return df_merged


    def _filter_dfs_common_tickers(self):
        # Initate list
        inner_tickers = []
        
        # Find tickers
        rh_tickers = self.df_rh.columns.values
        
        # Iterate over other df and find common
        for tick in self.df_crsp["ticker"].unique():
            if tick in rh_tickers:
                inner_tickers.append(tick)

        # Filter df
        self.df_crsp = self.df_crsp[self.df_crsp["ticker"].isin(inner_tickers) == True] 

        # Filter df keeping the sum column in the original df
        inner_tickers.append("sum") 
        df_copy = self.df_rh.copy()
        df_copy = df_copy.T
        df_copy = df_copy[df_copy.index.isin(inner_tickers) == True]

        self.df_rh = df_copy
        self.df_rh = self.df_rh.T

        return

    def _concat_users_to_robinhood(self):
        # Build Users df with Statista's data
        m_users = pd.DataFrame({
            "date": [
                pd.to_datetime("2017-12-31"),
                pd.to_datetime("2018-12-31"),
                pd.to_datetime("2019-12-31"),
                pd.to_datetime("2020-12-31")
            ],
            "users": [2, 6, 10, 12.5]
        })

        # Construct range to interpolate
        date_range = pd.date_range(start='2017-12-31', end='2020-12-31', freq='D')
        df_main = pd.DataFrame({'date': date_range})

        # Merge the two DataFrames
        df_combined = df_main.merge(m_users, on='date', how='left')

        # Interpolate missing annual values (linear interpolation)
        df_combined['users'] = df_combined['users'].interpolate(method='linear') # Change this for other types of interpolation
        df_combined.set_index("date", inplace=True)
        self.df_rh_long = self.df_rh_long.merge(df_combined, on='date', how='left')
        
        return
    

    # TODO: Make this an Analyzer method
    def compute_distances(self, df_merged:pd.DataFrame=None, n:int=100, by:str='mc_retail'):
        # Build merged df
        df_merged = self.merge_dfs()

        # Extract required columns first (avoids unnecessary memory usage)
        df = df_merged[['date', 'ticker', 'mc', "popularity", 'mc_retail', 'holders', 'prc_adj', 'shrcd', "vol"]].copy()

        # Sort by m in descending order (this is done for all dates at once)
        df_sorted = df.sort_values(by=['date', by], ascending=[True, False])

        # Build rank based on market cap and users
        df_sorted["rank_mkt"] = df_sorted.groupby("date")["mc"].rank(ascending=False)
        df_sorted["rank_ret"] = df_sorted.groupby("date")["mc_retail"].rank(ascending=False)
        df_sorted[f"rank_{by}"] = df_sorted.groupby("date")[by].rank(ascending=False)
        
        # Compute distance metric for rank
        df_sorted["rank_distance"] = (df_sorted["rank_mkt"] - df_sorted["rank_ret"]) / df_sorted["rank_ret"]

        # Compute distance metric for distribution
        df_sorted["m_daily"] = df_sorted[["date", "mc"]].groupby("date").transform('sum')
        df_sorted["mc_retail_daily"] = df_sorted[["date", "mc_retail"]].groupby("date").transform('sum')
        df_sorted[f"{by}_daily"] = df_sorted[["date", "mc"]].groupby("date").transform('sum')
        
        df_sorted["m_daily_pct"] = df_sorted["mc"] / df_sorted["m_daily"]
        df_sorted["mc_retail_daily_pct"] = df_sorted["mc_retail"] / df_sorted["mc_retail_daily"]
        df_sorted[f"{by}_daily_pct"] = df_sorted[by] / df_sorted[f"{by}_daily"]

        # Compute distance metric for rank
        df_sorted["pct_distance"] = (df_sorted["mc_retail_daily_pct"] - df_sorted["m_daily_pct"])*df_sorted["mc_retail_daily_pct"]
        df_sorted[f"{by}_pct_distance"] = (df_sorted[f"{by}_daily_pct"] - df_sorted["m_daily_pct"])*df_sorted[f"{by}_daily_pct"]

        # Keep only the top N tickers per date
        df_top_n = df_sorted.groupby('date').head(n)
        df_top_n = df_top_n.reset_index(drop=True)

        return df_top_n, df_sorted