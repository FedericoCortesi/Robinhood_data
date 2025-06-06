import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis

import statsmodels.stats.api as sms

import logging
from typing import List, Tuple

from . import setup_custom_logger
from .params import ReturnParams

# Setup logger
logger = setup_custom_logger(__name__, level=logging.INFO)



def log_ma_returns(
        levels:pd.DataFrame, 
        return_params: ReturnParams=None, 
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


def compute_crra_gamma(array:np.ndarray)->float:
    """
    Given the inputs it estimates the gamma of a CRRA utility function.
    
    Parameters:
    - array : np.ndarray
        Array of returns - risk free

    Returns:
    - gamma : float
    """
    mean = np.mean(array)
    var = np.var(array)

    gamma = (mean)/var + 1/2
    return gamma


def paramteric_expected_utility_crra(gamma:float, array:np.ndarray):
    """
    Given gamma and an array it computes the expected utility given a formula. The array should be of gross returns net of risk free.
    """
    gamma = float(gamma)

    log_returns = np.log(array)

    if gamma != 1:
        mu = np.mean(log_returns)
        s_2 = np.var(log_returns)

        numerator = np.exp(mu*(1-gamma) + (s_2/2)*(1-gamma)**2) - 1
        denominator = 1 - gamma

        utility =  numerator / denominator
    else:
        mu = np.mean(log_returns)
        utility =  mu

    #logger.debug(f"Gamma: {gamma:.4f} | Utility: {utility:.4f}")
    return utility


def error_ce(gamma:float, array_stocks:np.ndarray, array_benchmark:np.ndarray, parametric:bool=True):
    gamma = float(gamma)
    if parametric:
        u_stocks = paramteric_expected_utility_crra(gamma, array_stocks)
        u_becnhmark = paramteric_expected_utility_crra(gamma, array_benchmark)
    else:
        u_stocks = compute_crra_utility(array_stocks, gamma)
        u_becnhmark = compute_crra_utility(array_benchmark, gamma)

    error = np.abs(u_becnhmark - u_stocks) 


    logger.debug(f"Gamma: {gamma:.4f} | Error: {error:.4f} | Utility stocks: {u_stocks:.4f} | Utility Benchmark: {u_becnhmark:.4f}")
    return error


def find_gamma_certainty_equivalent_cutoff(array_stocks: np.ndarray, array_benchmark: np.ndarray, parametric:bool=True):
    result = minimize(
        error_ce,
        x0=np.array([1.01]),  # initial guess for gamma
        args=(array_stocks, array_benchmark, parametric),
        bounds=[(-20, 20)],  # set realistic bounds to ensure stable optimization
        method='Powell'
    )

    
    if result.success: 
        if round(result.fun, 3) == 0:
            gamma_hat = result.x[0]  
        else:
            logger.info(f"Didn't converge. Error: {result.fun}")
            gamma_hat = np.nan
    else:
        raise ValueError("Optimization failed:", result.message)

    return gamma_hat



def compute_crra_utility(returns_array:np.ndarray, gamma:float, confint:bool=False, alpha:float=0.05, mean:bool=True)->np.ndarray :
    """
    Computes the crra utility given a certain gross returns array and gamma.
    """
    # apply the limit
    if np.isclose(gamma, 1):
        utility_array = np.log(returns_array)

    # definition of CRRA
    else:
        utility_array = (returns_array**(1-gamma)-1)/(1-gamma)

    if not confint:
        if mean:
            return utility_array.mean()
        else:
            return utility_array.values

    else:
        ds = sms.DescrStatsW(utility_array)

        ci_low, ci_high = ds.tconfint_mean(alpha=alpha)

        if mean:
            return utility_array.mean(), ci_low, ci_high
        else: 
            return utility_array, ci_low, ci_high


def moment_condition_gmm(gamma: float,
                     rp: np.ndarray,
                     rf: np.ndarray) -> float:
    """
    Sample analogue of E[ (1/(1+rf_t))*(1+rp_t)^(-gamma)*(1+rp_t) ] - 1.
    
    Parameters
    ----------
    gamma : float
        CRRA risk aversion coefficient.
    rp : np.ndarray
        Realized excess returns array of shape (T,).
    rf : np.ndarray
        Risk-free rates array of shape (T,), corresponding to each return in rp.
        
    Returns
    -------
    float
        Sample moment: mean of discount factor times gross return minus one.
    """
    if rp.shape != rf.shape:
        raise ValueError("`rp` and `rf` must have the same shape.")
        
    df = 1.0 / (1.0 + rf)
    m_t = df * (1.0 + rp) ** (-gamma)
    return np.mean(m_t * (1.0 + rp)) - 1.0



def squared_moment(gamma: float,
                   rp: np.ndarray,
                   rf: np.ndarray) -> float:
    """
    Square of the moment condition, useful for minimization.
    """
    g = moment_condition_gmm(gamma, rp, rf)
    return g * g    


def gamma_closed_form(col:np.ndarray, alpha:float)-> float:
    """
    Computes the implied sample CRRA risk aversion from an array of simple returns using second order taylor approximation.
    """
    mu = col.mean()
    s2 = col.var()


    gamma = mu/(alpha*s2)
    return gamma

def crra_certainty_equivalent_utility(utility:float, gamma:float)->float:
    """
    Compute the certainty equivalent from a utility value and a risk aversion parameter
    """
    if np.isclose(gamma, 1):
        ce = np.exp(utility)
    else:
        ce = ((1-gamma)*utility+1)**(1/(1-gamma))

    return ce

def crra_ce_returns(returns:np.ndarray, gamma:float)->float:
    """
    Compute CRRA certainty equivalent for a given gross returns array and a gamma value
    
    ------
    Obtained by reversing the CRRA utility function and some math
    """


    # Apply definition (avoid division by zero)
    if not np.isclose(gamma, 1):
        
        # Apply transformation to returns 
        returns_modified = returns**(1-gamma)
        # Take mean
        returns_modified_bar = np.mean(returns_modified)

        # Apply formula 
        ce = returns_modified_bar**(1/(1-gamma))

    else:
        # Apply transformation to returns 
        returns_modified = np.log(returns) 
        # Take mean (Expectation) 
        returns_modified_bar = np.mean(returns_modified)
        # Adjust formula
        ce = returns_modified_bar + 1

    return ce 



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
