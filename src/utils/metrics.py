import pandas as pd
import numpy as np


def log_ma_returns(levels, horizons:list=[5,15,30, 60, 120]):
    """
    Given a dataframe with daily prices the function returns moving averages of log returns.
    """
    # Build returns df, applying logs ensures additivity of returns
    returns = np.log(levels / levels.shift(1)).fillna(0)
    returns.index = pd.to_datetime(returns.index)

    # Add the cumulative returns for the whole period and keep smaller values
    horizons.append(len(returns))
    horizons = [h for h in horizons if h <= len(returns)]
    horizons = sorted(list(set(horizons)))

    for d in horizons: # delete min_periods if you want to start at date d and not before
        for col in levels.columns:
            returns[f"{col}_{d}_return"] = returns[col].rolling(d, min_periods=1).sum()
        

    return returns, horizons

