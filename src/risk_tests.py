import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import bootstrap
from tqdm import trange   

import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple

from .utils.helpers import load_factors
from src.utils.metrics import find_gamma_certainty_equivalent, paramteric_expected_utility_crra, compute_crra_utility, squared_moment

# Setup logger
import logging
from .utils.custom_formatter import setup_custom_logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)

from . import Analyzer

class RiskTests():
    def __init__(self, analyzer:Analyzer=None):

        self.analyzer = analyzer
        self.mkt_index = self.analyzer.compare_tickers[0] # take the first

        self.factors = load_factors()
        self.factors.index = pd.to_datetime(self.factors.index, format="%Y%m%d")

        self._build_daily_factors()

        pass

    def _build_daily_factors(self):

        ret, _ = self.analyzer.build_returns()

        ret = (np.exp(ret) -1)*100 # get them in percentage returns

        self.factors = pd.DataFrame(self.factors["rf"]) # keep good columns and ensure is datafrme to merge later


        self.factors = self.factors.merge(ret[["rh_portfolio", self.mkt_index]], left_index=True, right_index=True, how="inner") # merge rh_portfolio
        self.factors = self.factors.rename(columns={self.mkt_index:"mkt"}) # rename for consistency
        self.factors /= 100 

        self.factors["xr"] = self.factors["rh_portfolio"] - self.factors["rf"] # take out risk free to have excess returns
        self.factors["xmkt"] = self.factors["mkt"] - self.factors["rf"] # 
        self.factors = self.factors[["rh_portfolio", "rf", "mkt", "xr", "xmkt"]]

        return self.factors
    
    def resample_factors(self, freq:str="M"):
        """
        Method to resample daily factors.
        """

        resampled_factors = (1+self.factors).resample(freq).prod() -1
        return resampled_factors



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

        obj    = lambda g: squared_moment(g, rp, bar_rf) # define object to minimize

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
        
    def find_cutoff_gamma(self, df_returns:pd.DataFrame=None, parametric:bool=False, plot:bool=True)->float:

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
        print(f"Gamma: {gamma_hat:.4f}")

        if plot:
            gap = 5

            #lower = gamma_hat - gap if gamma_hat < gamma_gmm else gamma_gmm - gap
            #upper = gamma_hat + gap if gamma_hat >= gamma_gmm else gamma_gmm + gap
            #lower = gamma_hat - gap 
            #upper = gamma_hat + gap 
            lower = gamma_hat - gap 
            upper = 10 

            # define grid to compute utilities
            gammas = np.linspace(lower, upper, 250)
            sns.set_style("whitegrid")

            if parametric:
                # compute utilities array
                utilities_rh_portfolio   = [paramteric_expected_utility_crra(g, df["xr"].values)   for g in gammas]
                utilities_xmkt = [paramteric_expected_utility_crra(g, df[mkt].values) for g in gammas]

                # plot
                plt.figure(figsize=(10, 6))
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
                plt.figure(figsize=(18,10))
                plt.plot(gammas, means_rh_portfolio, label='rh_portfolio mean utility', color='C0')
                plt.fill_between(gammas, lo_rh_portfolio, hi_rh_portfolio, color='C0', alpha=0.2)

                plt.plot(gammas, means_mk, label='MKT mean utility', color='C1')
                plt.fill_between(gammas, lo_mk, hi_mk, color='C1', alpha=0.2)

                plt.axvline(gamma_hat, color='red', linestyle='--', label=f'γ* = {gamma_hat:.3f}')
                #plt.axvline(gamma_gmm, color='purple', linestyle='--', label=f'γ found in GMM = {gamma_gmm:.3f}')
                plt.xlabel('Gamma')
                plt.ylabel('Expected utility')
                plt.title('Expected utility vs gamma with 95% CI')
                plt.legend()
                plt.grid(True)
                plt.show()
        
        return gamma_hat


    def bootstrap_difference_utility(self, df_returns:pd.DataFrame=None, gamma0:float=None)->Tuple[float, float, float]:
        """
        Bootstrap the differernce in utility for market and market around an initial gamma value.
        
        -------
        Returns:
            point_estimate
            lower
            upper
        """

        if df_returns is None:
            df_returns = self._build_daily_factors() +1
                    
        # parameters
        B = 5_000                   # number of bootstrap replications
        gamma0 = self.estimate_euler_gamma() if gamma0 is None else gamma0 # gamma around which you need to bootstrap difference
        n = len(df_returns)

        # storage for the bootstrap difference
        deltaU = np.empty(B)

        for b in trange(B):
            # 1) sample row‐indices with replacement
            idx = np.random.randint(0, n, size=n)
            df_b = df_returns.iloc[idx]

            # 2) compute mean utilities (no confint needed inside bootstrap)
            mu_rh = compute_crra_utility(df_b["xr"], gamma0, confint=False)
            mu_mk = compute_crra_utility(df_b["xmkt"], gamma0, confint=False)

            # 3) record difference: market − RH
            deltaU[b] = mu_mk - mu_rh

        # 4) build CI
        lower, upper = np.percentile(deltaU, [2.5, 97.5])
        point_estimate = np.mean(deltaU)

        print(f"ΔU (market - RH) at γ={gamma0:.3f}: "
            f"{point_estimate:.5f} "
            f"[{lower:.5f}, {upper:.5f}]")
        
        return (point_estimate, lower, upper)


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
                method='bca',
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