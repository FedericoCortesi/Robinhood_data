import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import bootstrap
from tqdm import trange, tqdm   

import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple, List

from .utils.helpers import load_factors, date_to_num
from src.utils.metrics import find_gamma_certainty_equivalent, paramteric_expected_utility_crra, compute_crra_utility, squared_moment
from config.constants import PROJECT_ROOT

# Setup logger
import logging
from .utils.custom_formatter import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

# Import other classes
from . import Analyzer
from src.utils.params import ReturnParams

class RiskTests():
    def __init__(self, analyzer:Analyzer=None, ff_mkt_index:bool=False, mkt_index:str=None, portfolio_column:str="rh_portfolio"):

        self.analyzer = analyzer

        if mkt_index is None:
            self.mkt_index = self.analyzer.compare_tickers[0] if len(self.analyzer.compare_tickers) > 0 else None # take the first
        else:
            assert mkt_index in self.analyzer.compare_tickers, f"mkt_index {mkt_index} not found in analyzer's tickers: {self.analyzer.compare_tickers}"
            self.mkt_index = mkt_index

        self.ff_mkt_index = ff_mkt_index

        self.portfolio_column = portfolio_column

        self.factors = load_factors()
        self.factors.index = pd.to_datetime(self.factors.index, format="%Y%m%d")

        self._build_daily_factors()

        pass

    def _build_daily_factors(self):

        ret, _ = self.analyzer.build_returns()

        ret = (np.exp(ret) -1)*100 # get them in percentage returns

        # keep good columns and ensure is datafrme to merge later 
        #self.factors = pd.DataFrame(self.factors[["rf", "xmkt", "pfioret100"]]) # suppressed to use factor in regressions

        cols_to_keep = [self.portfolio_column, self.mkt_index]
        cols_to_keep = [
            col for col in cols_to_keep
            if col is not None and col in ret.columns
        ]
         

        # handle case to use old returns
        if self.portfolio_column == "rh_portfolio":
            self.factors = self.factors.drop(columns=["pfioret100"])

        self.factors = self.factors.merge(ret[cols_to_keep], left_index=True, right_index=True, how="inner") # merge rh_portfolio
        
        if self.mkt_index and not self.ff_mkt_index:
            self.factors = self.factors.rename(columns={self.mkt_index:"mkt"}) # rename for consistency
        
        # find in decimal form
        self.factors /= 100 

        # take out risk free to have excess returns
        if self.portfolio_column in cols_to_keep:
            self.factors["xr"] = self.factors[self.portfolio_column] - self.factors["rf"] 
        
        # handle "market" index
        if self.mkt_index and not self.ff_mkt_index:
            self.factors["xmkt"] = self.factors["mkt"] - self.factors["rf"] # excess returns market
            #self.factors = self.factors[[self.portfolio_column, "rf", "mkt", "xr", "xmkt"]] # suppressed to use factor in regressions
        else:
            self.factors["mkt"] = self.factors["xmkt"] + self.factors["rf"] # normal returns market
            #cols_to_keep = [self.portfolio_column, "rf", "mkt", "xr", "xmkt"] # suppressed to use factor in regressions
            #cols_to_keep = [col for col in cols_to_keep if col in self.factors.columns] # suppressed to use factor in regressions
            #self.factors = self.factors[cols_to_keep] # suppressed to use factor in regressions

        return self.factors
    
    def resample_factors(self, freq:str="M"):
        """
        Method to resample daily factors.
        """

        resampled_factors = (1+self.factors).resample(freq).prod() -1
        return resampled_factors

    def _build_all_windows(self, df:pd.DataFrame)->List:
        # Set doesnt allow to duplicate items
        all_windows = set()

        # get dates to iterate on
        dates = df.index.unique()

        for date1 in dates:
            for date2 in dates:
                # Order the dates so that i dont have problems in computing reteurns and i dont have tuples with the same value in different order 
                if date2>date1:
                    
                    inner_tuple = (date1, date2)
                    all_windows.add(inner_tuple)

        # easier to handle
        all_windows = list(all_windows)    
        return all_windows    


    def build_all_pairs_dataframe(self, df_returns:pd.DataFrame=None)->pd.DataFrame:
        """
        Build the dataframe of all gross excess returns for all possible windows
        """
        if df_returns is None:
            df_returns = self.factors.copy()

        # compute levels as cumulative returns, it is like setting W = 1 @ time=0
        levels = (df_returns+1).cumprod()
        
        all_windows = self._build_all_windows(df=levels)

        all_ret = []
        for window in tqdm(all_windows):

            # get first and last value (days)
            beg = window[0]
            end = window[1]
            
            # Compute days between them

            # compute gross returns
            ret = levels.loc[end]/levels.loc[beg]

            all_ret.append(ret)


        # Create DataFrame from the list of Series
        all_ret_df = pd.DataFrame(all_ret)

        # Assign MultiIndex with start and end dates
        all_ret_df.index = pd.MultiIndex.from_tuples(all_windows, names=["start_date", "end_date"])

        all_ret_df['date_difference'] = all_ret_df.index.map(lambda x: (x[1] - x[0]).days)

        return all_ret_df

    def estimate_euler_gamma(self, df_returns:pd.DataFrame=None, portfolio_name:str="rh_portfolio"):
        """
        Uses the euler condition on CRRA utility and root finding to estimate the  risk aversion of the portfolio
        """

        # Load data
        if df_returns is None:
            df_gmm = self.factors.copy()
        else:
            df_gmm = df_returns

        rp = df_gmm[portfolio_name]  # daily portfolio returns as decimals (e.g., 0.001 = 0.1%)
        ret_f = df_gmm['rf']  # daily risk-free rate as decimals

        # Compute average daily risk-free rate
        bar_rf = ret_f.mean()

        obj    = lambda g: squared_moment(g, rp, ret_f) # define object to minimize

        # Minimize it over a sensible interval
        #result = minimize_scalar(squared_moment_local, bounds=(-50, 50), method='bounded')
        result = minimize_scalar(obj, bounds=(-50, 50), method='bounded')

        if result.success:
            #print(f"Estimated CRRA coefficient γ (approximate GMM): {result.x:.4f}")
            gamma_gmm = result.x
            return gamma_gmm
        else:
            print("Minimization failed. Try inspecting the function or using a wider interval.")
            return np.nan
        
    def find_cutoff_gamma(self, df_returns:pd.DataFrame=None, parametric:bool=False, plot:bool=True, bounds:tuple=None)->float:

        # Load data
        if df_returns is None:
            df_returns = self.factors.copy()


        df = df_returns.copy() + 1
        mkt = "xmkt"


        # keep the scalar returned by the CE condition
        gamma_hat = find_gamma_certainty_equivalent(df["xr"].dropna(),
                                                    df[mkt].dropna(),
                                                    parametric=parametric)


        gamma_hat = gamma_hat if gamma_hat is not np.nan else 0
        print(f"Gamma: {gamma_hat:.5f}")

        if plot:

            # define grid to compute utilities
            #lower = gamma_hat - gap if gamma_hat < gamma_gmm else gamma_gmm - gap
            #upper = gamma_hat + gap if gamma_hat >= gamma_gmm else gamma_gmm + gap
            if bounds is None:
                gap = 5
                lower = gamma_hat - gap 
                upper = gamma_hat + gap 
                gammas = np.linspace(lower, upper, 250)
            
            else:
                assert (bounds[0] <= gamma_hat) and (bounds[1] >= gamma_hat), f"Gamma not in bounds: {bounds}"
                gammas = np.linspace(bounds[0], bounds[1], 250)
                

            sns.set_style("whitegrid")

            if parametric:
                # compute utilities array
                utilities_rh_portfolio   = [paramteric_expected_utility_crra(g, df["xr"].values)   for g in gammas]
                utilities_xmkt = [paramteric_expected_utility_crra(g, df[mkt].values) for g in gammas]

                # plot
                plt.figure(figsize=(18, 12))
                plt.plot(gammas, utilities_rh_portfolio,   label='Portfolio')
                plt.plot(gammas, utilities_xmkt, label='Market')
                plt.axvline(gamma_hat, color='red', linestyle='--', alpha=0.6,
                            label=f'γ* = {gamma_hat:.3f}')
                plt.xlabel('Gamma (relative risk-aversion parameter)')
                plt.ylabel('Expected CRRA utility')
                plt.title('Expected utility as a function of gamma')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                # compute utilities array with confidence intervals
                utilities_rh_portfolio   = [compute_crra_utility(df["xr"],   g, confint=True, alpha=0.05) for g in gammas]
                utilities_xmkt = [compute_crra_utility(df[mkt], g, confint=True, alpha=0.05) for g in gammas]

                # Make each obs a tuple
                means_rh_portfolio, lo_rh_portfolio, hi_rh_portfolio = zip(*utilities_rh_portfolio)
                means_mk, lo_mk, hi_mk = zip(*utilities_xmkt)

                # plot
                # create figure+axes
                fig, ax = plt.subplots(figsize=(18, 10))

                # plot RH portfolio
                ax.plot(gammas, means_rh_portfolio, label='RH Portfolio', color='C0')
                ax.fill_between(gammas, lo_rh_portfolio, hi_rh_portfolio, color='C0', alpha=0.2)

                # plot market
                ax.plot(gammas, means_mk,     label='Market',       color='C1')
                ax.fill_between(gammas, lo_mk, hi_mk,              color='C1', alpha=0.2)

                # vertical line for gamma_hat
                ax.axvline(gamma_hat, color='red', linestyle='--',
                        label=f'γ* = {gamma_hat:.3f}')

                # labels
                ax.set_xlabel('Gamma')
                ax.set_ylabel('Expected Utility')

                # **set title on the axes** with explicit fontsize
                fig.suptitle('Expected utility vs gamma with 95% CI', fontsize=18)

                ax.legend(fontsize=14)
                ax.grid(True)
                plt.tight_layout()
                plt.show()

        return gamma_hat

    def bootstrap_difference_utility(self, df_returns=None, gamma0=None, portfolio_name="rh_portfolio", 
                                    market_name="xmkt", n_resamples=5000, confidence_level=0.95, 
                                    print_results=True) -> Tuple[float, float, float]:
        """
        Bootstrap the difference in utility between market portfolio and another portfolio using scipy's bootstrap.
        
        Parameters:
        -----------
        df_returns : pd.DataFrame, optional
            DataFrame containing portfolio returns and market returns.
            If None, will use internal daily factors.
        gamma0 : float, optional
            Risk aversion parameter to use for utility calculations.
            If None, will estimate using estimate_euler_gamma().
        portfolio_name : str, default="rh_portfolio"
            Column name for the portfolio returns in the DataFrame.
        market_name : str, default="xmkt"
            Column name for the market returns in the DataFrame.
        n_resamples : int, default=5000
            Number of bootstrap resamples to generate.
        confidence_level : float, default=0.95
            Confidence level for the interval (between 0 and 1).
        print_results : bool, default=True
            Whether to print the results to console.
            
        Returns:
        --------
        Tuple[float, float, float]
            Contains (point_estimate, lower_bound, upper_bound) for the utility difference.
        
        Notes:
        ------
        Computes the difference in CRRA utility between the market portfolio and the specified portfolio
        using scipy's bootstrap to estimate confidence intervals. The difference is calculated as market - portfolio.
        """
        from scipy.stats import bootstrap
        
        # Input validation
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive")
        
        # Load data if not provided
        if df_returns is None:
            df_returns = self._build_daily_factors() + 1  # Adding 1 as in the original
        
        # Ensure the required columns exist
        required_cols = [portfolio_name, market_name]
        if not all(col in df_returns.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_returns.columns]
            raise ValueError(f"Missing required columns in df_returns: {missing}")
        
        # Estimate gamma if not provided
        if gamma0 is None:
            gamma0 = self.estimate_euler_gamma()
        
        # Extract returns as numpy arrays
        portfolio_returns = df_returns[portfolio_name].values
        market_returns = df_returns[market_name].values
        
        # Compute the point estimate on the full data
        mu_portfolio = compute_crra_utility(portfolio_returns, gamma0, confint=False)
        mu_market = compute_crra_utility(market_returns, gamma0, confint=False)
        point_estimate = mu_market - mu_portfolio
        
        try:
            # Define the statistic function with compatible signature for scipy bootstrap
            def utility_diff_bootstrap(rp, mr):
                """Bootstrap statistic function that accepts the format scipy provides"""
                # Extract portfolio and market returns from data_array
                portfolio_rets = rp  # First element is portfolio returns
                market_rets = mr     # Second element is market returns
                
                # Compute mean utilities
                mu_portfolio = compute_crra_utility(portfolio_rets, gamma0, confint=False)
                mu_market = compute_crra_utility(market_rets, gamma0, confint=False)
                
                # Return difference: market - portfolio
                return mu_market - mu_portfolio
            
            # Prepare data for bootstrap
            data = (portfolio_returns, market_returns)
            
            # Run bootstrap using scipy's bootstrap function
            res = bootstrap(
                data,
                statistic=utility_diff_bootstrap,
                paired=True,
                vectorized=False,  # Set to False since our statistic works on a single sample
                n_resamples=n_resamples,
                method='bca',      # Using BCa method as requested
                confidence_level=confidence_level
            )
            
            # Extract confidence interval
            ci = res.confidence_interval
            lower, upper = ci.low, ci.high
            
            if print_results:
                print(f"ΔU ({market_name} - {portfolio_name}) at γ={gamma0:.3f}: "
                    f"{point_estimate:.5f} "
                    f"[{lower:.5f}, {upper:.5f}]")
            
            return (point_estimate, lower, upper)
        
        except Exception as e:
            print(f"Bootstrap failed with error: {str(e)}")
            # Return NaN values if bootstrap fails
            return (float('nan'), float('nan'), float('nan'))        



    def bootstrap_gamma(self, df_returns=None, portfolio_name="rh_portfolio", n_resamples=5000, 
                    confidence_level=0.95, print_results=True):
        """
        Performs bootstrap confidence interval estimation for the CRRA risk aversion parameter.
        
        Parameters:
        -----------
        df_returns : pd.DataFrame, optional
            DataFrame containing portfolio returns and risk-free rate. 
            If None, will use internal daily factors.
        portfolio_name : str, default="rh_portfolio"
            Column name for portfolio returns in the DataFrame
        n_resamples : int, default=5000
            Number of bootstrap resamples to generate
        confidence_level : float, default=0.95
            Confidence level for the interval (between 0 and 1)
        print_results : bool, default=True
            Whether to print the results to console
            
        Returns:
        --------
        dict
            Contains point estimate and confidence interval bounds
        
        Notes:
        ------
        Uses the BCa (bias-corrected and accelerated) bootstrap method to estimate
        confidence intervals for the CRRA risk aversion parameter.
        """
        # Input validation
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if n_resamples <= 0:
            raise ValueError("n_resamples must be positive")
        
        # Load data if not provided
        if df_returns is None:
            df_returns = self._build_daily_factors()

        # Ensure the required columns exist
        required_cols = [portfolio_name, "rf"]
        if not all(col in df_returns.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_returns.columns]
            raise ValueError(f"Missing required columns in df_returns: {missing}")

        # Full-sample estimate
        gamma_hat = self.estimate_euler_gamma(df_returns, portfolio_name)
        
        if print_results:
            print(f"γ̂^GMM = {gamma_hat:.4f}")

        # Prepare data for bootstrap
        rp = df_returns[portfolio_name]
        rf = df_returns["rf"]
        data = (rp, rf)

        # Statistic function that complies with bootstrap parameters
        def gmm_stat(rp, rf):
            """Wrapper around estimate_euler_gamma to comply with bootstrap requirements"""
            # Create temporary DataFrame with bootstrap sample
            bootstrap_data = pd.DataFrame({
                portfolio_name: rp,
                "rf": rf
            })
            # Use existing estimation method
            return self.estimate_euler_gamma(df_returns=bootstrap_data, portfolio_name=portfolio_name)

        try:
            # Run BCa bootstrap
            res = bootstrap(
                data,
                statistic=gmm_stat,
                paired=True,
                vectorized=False,
                n_resamples=n_resamples,
                method='BCa',
                confidence_level=confidence_level
            )

            ci = res.confidence_interval
            
            if print_results:
                print(f"{confidence_level*100:.0f}% BCa CI = [{ci.low:.4f}, {ci.high:.4f}]")
            
            # Return results as a dictionary
            results = {
                'gamma_estimate': gamma_hat,
                'ci_lower': ci.low,
                'ci_upper': ci.high,
                'confidence_level': confidence_level,
                'n_resamples': n_resamples
            }
            
            return results
            
        except Exception as e:
            print(f"Bootstrap failed with error: {str(e)}")
            # Return partial results if bootstrap fails
            return {
                'gamma_estimate': gamma_hat,
                'error': str(e)
            }
        

    def run_regressions_factor_models(self, df_returns:pd.DataFrame=None, factors:int=0):
        """
        Run factor models on df_returns.

        Parameters:
        -----------
            df_returns: optional, dataframe with returns 
            factors: int, must be 0, 1 or 6

        Returns:
        ---- 
        """
        # import only here because its heavy 
        import statsmodels.api as sm

        # define df if absent
        if df_returns is None:
            df_returns = self.factors

        assert factors in [0, 1, 6], f"Number of factors must be in {[0, 1, 6]}"    

        # ectract regressors col
        ordered_factors = ['xmkt', 'smb', 'hml', 'rmw', 'cma', 'umd']
        regressors = ordered_factors[:factors]

        X = df_returns[regressors]
        y = df_returns["xr"]

        X = sm.add_constant(X)  # this adds the alpha term

        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())



