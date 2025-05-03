## comparison_1.png

from src import Analyzer
from src.utils.params import ReturnParams

return_params = ReturnParams(
    horizons={5, 30, 120},  
    start_date=None,
    cumulative=True,
    append_start=False
)

an_fedyk = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=True, weights_application="wealth", dl_kwargs={"weights_method":"dollar"})
an_mine = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=True, weights_application="number", dl_kwargs={"weights_method":"dollar"})

an_compare = Analyzer(compare_tickers=["VT", "VOO"], exclude_rh=True, return_params=return_params, stocks_only=False, weights_application="wealth")

r_fedyk, h_fedyk = an_fedyk.build_returns()
r_mine, h_mine = an_mine.build_returns()
r_compare, h_compare = an_compare.build_returns()

from src.visuals.plotter import Plotter, ReturnSeries

plotter = Plotter([
    ReturnSeries(label="Fedyk", df=r_fedyk, horizons=h_fedyk),
    ReturnSeries(label="Mine", df=r_mine, horizons=h_mine),
    ReturnSeries(label="", df=r_compare, horizons=h_compare)
])

plotter.plot_returns_timeseries(custom_labels=["Fedyk", "Mine", "World ETF", "S&P 500"], save=True, file_name="returns/comparison_1.png")
plotter.plot_returns_kdes(custom_labels=["Fedyk", "Mine", "World ETF", "S&P 500"], save=True, file_name="distributions/comparison_1.png")

## comparison_2.png

from src import Analyzer
from src.utils.params import ReturnParams

return_params = ReturnParams(
    horizons={5, 30, 120},  
    start_date=None,
    cumulative=True,
    append_start=False
)

an_fedyk = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="wealth", dl_kwargs={"weights_method":"dollar"})
an_mine = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="number", dl_kwargs={"weights_method":"dollar"})

an_compare = Analyzer(compare_tickers=["VT", "VOO"], exclude_rh=True, return_params=return_params, stocks_only=False, weights_application="wealth")

r_fedyk, h_fedyk = an_fedyk.build_returns()
r_mine, h_mine = an_mine.build_returns()
r_compare, h_compare = an_compare.build_returns()

from src.visuals.plotter import Plotter, ReturnSeries

plotter = Plotter([
    ReturnSeries(label="Fedyk", df=r_fedyk, horizons=h_fedyk),
    ReturnSeries(label="Mine", df=r_mine, horizons=h_mine),
    ReturnSeries(label="", df=r_compare, horizons=h_compare)
])

plotter.plot_returns_timeseries(custom_labels=["Fedyk", "Mine", "World ETF", "S&P 500"], save=True, file_name="returns/comparison_2.png")


## st_all.png

from src import Analyzer
from src.utils.params import ReturnParams

return_params = ReturnParams(
    horizons={1},  
    start_date=None,
    cumulative=False,
    append_start=False
)

an_fedyk = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="wealth", dl_kwargs={"weights_method":"dollar"})
an_mine = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="number", dl_kwargs={"weights_method":"dollar"})

an_compare = Analyzer(compare_tickers=["VT", "VOO"], exclude_rh=True, return_params=return_params, stocks_only=False, weights_application="wealth")

r_fedyk, h_fedyk = an_fedyk.build_returns()
r_mine, h_mine = an_mine.build_returns()
r_compare, h_compare = an_compare.build_returns()

from src.visuals.plotter import Plotter, ReturnSeries

plotter = Plotter([
    ReturnSeries(label="Fedyk", df=r_fedyk, horizons=h_fedyk),
    ReturnSeries(label="Mine", df=r_mine, horizons=h_mine),
    ReturnSeries(label="", df=r_compare, horizons=h_compare)
])

plotter.plot_returns_timeseries(custom_labels=["Fedyk", "Mine", "World ETF", "S&P 500"], save=True, file_name="/returns/st_all.png")
plotter.plot_returns_kdes(custom_labels=["Fedyk", "Mine", "World ETF", "S&P 500"], save=True, file_name="/distributions/st_all.png")

# cutoff_daily.png
from src import Analyzer
from src import RiskTests
from src.utils.params import ReturnParams

import numpy as np

return_params = ReturnParams(
    horizons={},  
    start_date=None,
    cumulative=False,
    append_start=False
)

an = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="wealth", dl_kwargs={"weights_method":"dollar"})


rt = RiskTests(an)
daily_factors = rt.factors
rt.find_cutoff_gamma()


# cutoff_number_all.png
from src import Analyzer
from src import RiskTests
from src.utils.params import ReturnParams

return_params = ReturnParams(
    horizons={},  
    start_date=None,
#    end_date="2020-02-03",
    cumulative=False,
    append_start=False
)

an = Analyzer(compare_tickers=[], return_params=return_params, stocks_only=False, weights_application="number", dl_kwargs={"weights_method":"dollar"})


rt = RiskTests(an)

all_ret_df = rt.build_all_pairs_dataframe()
all_ret_df.describe()
rt.find_cutoff_gamma(df_returns=all_ret_df-1, bounds=(-8,2))
