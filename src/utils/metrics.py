import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis

import logging
from typing import List, Tuple

from . import setup_custom_logger
from .params import ReturnParams

# Setup logger
logger = setup_custom_logger(__name__, level=logging.DEBUG)



def log_ma_returns(
        levels:pd.DataFrame, 
        return_params: ReturnParams, 
        returns_columns:List[str]=[]) -> Tuple[pd.DataFrame, list]:
    """
    Given a dataframe with daily prices the function returns moving averages of log returns and a list of the computed horizons.

    Parameters
    ----------
    levels : pd.DataFrame, df of values of the portfolio and other securities
    return_params : ReturnParams, ReturnParams object with the parameters needed to build returns  
    return_columns : str, name of the column with daily returns (if any)  

    Returns
    -------
    returns : pd.DataFrame, df of returns of the portfolio 
    horizons : list, all horizons used to compute returns 
    
    """
    logger.debug(f"returns columns: {returns_columns}")

    # Default rp
    return_params = return_params if return_params is not None else ReturnParams()

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
    cumulative = return_params.cumulative
    horizons = return_params.horizons
    if cumulative:
        horizons.add(len(returns))
    
    # Ensure its bounded and sorted
    horizons = return_params.horizons
    horizons = [h for h in horizons if h <= len(returns)]
    horizons = sorted(list(set(horizons)))

    append_start = return_params.append_start
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

def test_first_order_stochastic_dominance(series_a:pd.Series, series_b:pd.Series):
    """
    Test for first-order stochastic dominance between two return series from a DataFrame.
    
    Parameters:
    - series_a : pd.Series
        First return series
    - series_a : pd.Series
        Second return series
        
    Returns:
    - tuple:
        - dominance: bool, True if series A dominates series B
        - cdf_a_interp: numpy.array, interploated CDF of series A
        - cdf_b_interp: numpy.array, interploated CDF of series B
        - x_grid: numpy.array, common x-axis values for the integrated CDFs
        - dominance_confidence: float, percentage of points where the dominance relation holds
    """    

    # Extract and drop NaN values
    returns_a = series_a.dropna().values
    returns_b = series_b.dropna().values
            
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
        
    # Test for SSD: A dominates B if integrated_cdf_a <= integrated_cdf_b for all points
    dominance_points = cdf_a_interp <= cdf_b_interp
    dominance = np.all(dominance_points)
    dominance_confidence = np.mean(dominance_points) * 100
    
    return dominance, cdf_a_interp, cdf_b_interp, x_grid, dominance_confidence


def test_second_order_stochastic_dominance(series_a:pd.Series, series_b:pd.Series):
    """
    Test for second-order stochastic dominance between two return series from a DataFrame.
    
    Parameters:
    - series_a : pd.Series
        First return series
    - series_a : pd.Series
        Second return series
        
    Returns:
    - tuple:
        - dominance: bool, True if series A dominates series B
        - integrated_cdf_a: numpy.array, integrated CDF values for series A
        - integrated_cdf_b: numpy.array, integrated CDF values for series B
        - x_grid: numpy.array, common x-axis values for the integrated CDFs
        - dominance_confidence: float, percentage of points where the dominance relation holds
    """    

    # Extract and drop NaN values
    returns_a = series_a.dropna().values
    returns_b = series_b.dropna().values
            
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
