from src import Analyzer

an = Analyzer(compare_tickers=["TSLA"])

an.plot_returns_timeseries(save=False)
#an.plot_returns_kdes(save=True)