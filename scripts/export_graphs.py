from src import Analyzer

an = Analyzer(compare_tickers=["VOO", "VT"])

all_params = {
    "all":{"horizons":[1, 5], "start_date":None, "cumulative":False},
    "before":{"horizons":[1, 5], "start_date":None, "end_date":"2020-02-03", "cumulative":False},
    "after":{"horizons":[1, 5], "start_date":"2020-02-03", "cumulative":False}}


for name, params in all_params.items():
    an.return_params = params
    an.plot_returns_timeseries(save=True, name=f"ts_{name}_1_5.png", show=False)
    #an.plot_returns_kdes(save=True, name=f"kdes_{name}_1_5.png", show=False)
    #an.plot_returns_cdfs(save=True, name=f"cdfs_{name}_1_5.png", show=False)

print("Done!")