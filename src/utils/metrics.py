import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis

import logging
from . import setup_custom_logger

# Setup logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)



def log_ma_returns(levels:pd.DataFrame, horizons:set={5,15,30, 60, 120}, cumulative:bool=True, append_start:bool=True, returns_columns:list=[]):
    """
    Given a dataframe with daily prices the function returns moving averages of log returns and a list of the computed horizons.
    """
    # Initialize returns
    returns = levels.copy()

    # Iterate over columns in returns_columns
    for col in levels.columns:
        # check if returns are already present
        if col not in returns_columns:
            # Build returns df, applying logs ensures additivity of returns
            returns[col] = np.log(levels[col] / levels[col].shift(1)).fillna(0)
        else:
            returns[col] = levels[col]

    # ensure datetime format
    returns.index = pd.to_datetime(returns.index)

    # Add the cumulative returns for the whole period and keep smaller values
    if cumulative:
        horizons.add(len(returns))
    
    # Ensure its bounded and sorted
    horizons = [h for h in horizons if h <= len(returns)]
    horizons = sorted(list(set(horizons)))

    for h in horizons: # delete min_periods if you want to start at date d and not before
        min_periods = 1 if append_start else h
        min_periods = min_periods if min_periods < len(returns) else 1
        for col in levels.columns:
            returns[f"{col}_{h}_return"] = returns[col].rolling(h, min_periods=min_periods).sum()

    return returns, horizons


def compute_crra_gamma_fourth_moment(returns: np.ndarray, risk_free: float = 0) -> float:
    """
    Estimates the risk aversion coefficient (gamma) in a CRRA utility function
    using a higher-order Taylor approximation of expected utility.

    Parameters:
    - returns: np.ndarray of portfolio returns (gross or net)
    - risk_free: risk-free rate (same scale as returns)

    Returns:
    - Estimated gamma that maximizes expected utility
    """
    # cleam array from nans
    returns = returns[~np.isnan(returns)]

    # Compute empirical moments
    mu = np.mean(returns)
    sigma2 = np.var(returns, ddof=1)
    mu3 = skew(returns, bias=False) * (sigma2 ** 1.5)
    mu4 = kurtosis(returns, bias=False, fisher=False) * (sigma2 ** 2)  # normal kurt = 3

    def expected_utility(gamma: float) -> float:
        """
        Approximated expected utility under CRRA preferences with 4th-order Taylor expansion.
        """
        if gamma == 1:
            # Log utility case
            return np.mean(np.log(returns))
        
        term1 = 1 / (1 - gamma)
        term2 = -0.5 * gamma * (gamma + 1) * sigma2 / mu**2
        term3 = (1/6) * gamma * (gamma + 1) * (gamma + 2) * mu3 / mu**3
        term4 = - (1/24) * gamma * (gamma + 1) * (gamma + 2) * (gamma + 3) * mu4 / mu**4
        
        return mu**(1 - gamma) * (term1 + term2 + term3 + term4)

    # Objective: negative expected utility (since we minimize)
    objective = lambda gamma: -expected_utility(gamma)

    # Optimize gamma in a reasonable range (e.g., 0.01 to 15)
    result = minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    if result.success:
        logger.debug("success")

    return result.x if result.success else np.nan


def compute_crra_utility(returns_array:np.ndarray, gamma:float)->np.ndarray :
    """
    Computes the crra utility given a certain gross returns array and gamma.
    """
    # apply the limit
    if np.isclose(gamma, 1):
        utility_array = np.log(returns_array)

    # definition of CRRA
    else:
        utility_array = (returns_array**(1-gamma)-1)/(1-gamma)
    
    return utility_array




