import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Analyzer import Analyzer

an = Analyzer()


df_merged = an.df_merged


levels_rh = an._build_levels()

all_windows = set()

dates = df_merged["date"].unique()

for date1 in dates:
    for date2 in dates:
        if date2>date1:
            inner_tuple = (date1, date2)
            all_windows.add(inner_tuple)

all_windows = list(all_windows)


levels_rh = levels_rh/levels_rh.iloc[0]

all_ret = []
for window in tqdm(all_windows):
    beg = window[0]
    end = window[1]
    
    time = end - beg
    days = time.days

    ret = (levels_rh.loc[end]/levels_rh.loc[beg])**(1/days) - 1

    all_ret.append(ret)



# Create DataFrame from the list of Series
all_ret_df = pd.DataFrame(all_ret)

# Assign MultiIndex with start and end dates
all_ret_df.index = pd.MultiIndex.from_tuples(all_windows, names=["start_date", "end_date"])

def crra(col, gamma):
    utility = (col**(1-gamma)-1)/(1-gamma)
    return utility

utility = crra(all_ret_df, gamma=2)

# Compute quantiles
lower = utility.quantile(0.005)
upper = utility.quantile(0.995)

# Filter out rows where any column is outside the 1st–99th percentile range
all_ret_df_clean = utility[
    ~((utility < lower) | (utility > upper)).any(axis=1)
]

# Plot for each column
for col in utility.columns:
    data = np.sort(utility[col])
    cdf = np.arange(1, len(data) + 1) / len(data)
    plt.plot(data, cdf, label=col)

plt.title("Empirical CDF of Returns")
plt.xlabel("Return")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.show()

