from src.Analyzer import Analyzer

an = Analyzer(compare_tickers=["VOO", "VT"])

an.plot_returns_timeseries(save=True)
an.plot_returns_kdes(save=True)