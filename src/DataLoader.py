import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class DataLoader:
    def __init__(self, df_robinhood_path:str="df_rh.csv", df_crsp_path:str="df_crsp.csv", handle_nans:str="keep"):
        # load variables
        self.df_robinhood_path = os.path.join(ROOT, 'Data', df_robinhood_path)
        self.df_crsp_path = os.path.join(ROOT, 'Data', df_crsp_path)

        assert handle_nans in ["fill", "drop", "keep"], "only 'fill', 'drop', and 'keep' are possible values for handle_nans"
        self.handle_nans = handle_nans

        # Load Dataframes
        self._load_robinhood_data()
        self._load_crsp_data()

    def _load_robinhood_data(self):
        """"
        Loads the dataframe of all available robinhood securities.
        Nans are filled with zero and then deleted because some stocks have zeros instead of nans in the original data.
        """
        print("Loading Robinhood data")
        
        # Load csv
        df_rh = pd.read_csv(self.df_robinhood_path, index_col=0, parse_dates=[0])

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

        # Count the number of non-nan stocks
        df_rh.loc[:,"num_stocks"] = len(df_rh.columns) - df_rh.isna().sum(axis=1)

        # Save
        if self.handle_nans == "fill":
            df_rh = df_rh.bfill(axis=0)
            df_rh = df_rh.ffill(axis=0)
        elif self.handle_nans == "drop":
            df_rh = df_rh.dropna(axis=1)
        else:
            pass
        self.df_rh = df_rh
        return
    
    def _load_crsp_data(self):
        """
        Loads CRSP Data
        """
        print("Loading CRSP data")

        # Load csv
        df_crsp = pd.read_csv(self.df_crsp_path, index_col=[0], parse_dates=[1])

        if False:
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
            df_crsp.loc[:, "prc"] = [-x if x < 0 else x for x in df_crsp["prc"].to_list()]

            # Adjust price and shares outstanding 
            df_crsp.loc[:,"prc_adj"] = df_crsp["prc"]/df_crsp["cfacpr_adj"] 
            df_crsp.loc[:,"shrout_adj"] = df_crsp["shrout"]*1000*df_crsp["cfacshr_adj"] 

            # Drop other variables
            df_crsp = df_crsp.drop(columns=["cfacpr_last", "cfacshr_last", "facpr", "facshr"])
            
            # Prune eliminate instances that don't have shares outstanding
            df_crsp = df_crsp[df_crsp["shrout_adj"]!=0]

        # Save
        df_crsp = df_crsp.drop(columns=["cfacshr_adj", "cfacpr_adj", "cfacshr", "cfacpr", "sprtrn"])
        self.df_crsp = df_crsp
        
        return
    
    def _build_df_crsp(self):
        # Load csv
        df_crsp = pd.read_csv("./Data/tickers_volume.csv", index_col=[0], parse_dates=[1])

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
        df_crsp.loc[:, "prc"] = [-x if x < 0 else x for x in df_crsp["prc"].to_list()]

        # Adjust price and shares outstanding 
        df_crsp.loc[:,"prc_adj"] = df_crsp["prc"]/df_crsp["cfacpr_adj"] 
        df_crsp.loc[:,"shrout_adj"] = df_crsp["shrout"]*1000*df_crsp["cfacshr_adj"] 

        # Drop other variables

        # Prune eliminate instances that don't have shares outstanding
        df_crsp = df_crsp[df_crsp["shrout_adj"]!=0]

        return df_crsp
    

    def merge_dfs(self, users:bool=False, start_date:str=None):
        print("Merging...")
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

        if start_date:
            self.df_rh_long = self.df_rh_long[self.df_rh_long["date"]>=start_date]
            self.df_crsp = self.df_crsp[self.df_crsp["date"]>=start_date]

        # Merge both dataframes on 'date' and 'ticker'
        df_merged = self.df_rh_long.merge(self.df_crsp, on=['date', 'ticker'], how='inner')

        # Get the total number of trading days and filter out tickers with different values (missing or repeating data)
        trading_days = df_merged["date"].nunique()
        tickers_to_keep = df_merged["ticker"].value_counts()[df_merged["ticker"].value_counts()==trading_days].index
        df_merged = df_merged[df_merged["ticker"].isin(tickers_to_keep)]

        # Drop columns
        df_merged = df_merged.drop(columns=["prc", "shrout", "permno"])


        # Build additional features
        df_merged["daily_returns"] = df_merged.groupby("ticker")["prc_adj"].apply(lambda x: np.log(x / x.shift(1))).reset_index(level=0, drop=True).fillna(0)
        df_merged["cumulative_returns"] = df_merged.groupby("ticker")["daily_returns"].cumsum()
        df_merged['mc'] = df_merged['prc_adj'] * df_merged['shrout_adj']
        df_merged['mc_retail'] = df_merged['prc_adj'] * df_merged["holders"]
        #df_merged["retail_ownership"] = df_merged["holders"] / df_merged["shrout_adj"]
        df_merged["holders_change_pct"] = df_merged.groupby("ticker")["holders"].pct_change()
        df_merged["holders_change_diff"] = df_merged.groupby("ticker")["holders"].diff()
        df_merged["total_holders"] = df_merged[["date", "holders"]].groupby("date").transform("sum")
        df_merged["popularity"] = df_merged["holders"] / df_merged[["date", "holders"]].groupby("date")["holders"].transform("sum")
        df_merged['market_weight'] = df_merged['mc'] / df_merged[["date", "mc"]].groupby("date")["mc"].transform("sum")
        df_merged['retail_weight'] = df_merged['mc_retail'] / df_merged[["date", "mc_retail"]].groupby("date")["mc_retail"].transform("sum")

        # Create total holders change
        total_holders_series = df_merged.drop_duplicates("date")[["date", "total_holders"]].set_index("date").sort_index()
        total_holders_change_pct = total_holders_series.pct_change().rename(columns={"total_holders": "total_holders_change_pct"})
        total_holders_change_diff = total_holders_series.diff().rename(columns={"total_holders": "total_holders_change_diff"})

        # Merge back to df_merged
        df_merged = df_merged.merge(total_holders_change_pct, on="date", how="left")
        df_merged = df_merged.merge(total_holders_change_diff, on="date", how="left")

        # Sort
        df_merged = df_merged.sort_values(by=["ticker", "date"])
        df_merged = df_merged.reset_index(drop=True)

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
    

    def compute_distances(self, df_merged:pd.DataFrame=None, n:int=100, by:str='mc_retail'):
        if not hasattr(self, "df_merged"):  # Check if attribute exists
            self.merge_dfs()
        else:
            pass
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