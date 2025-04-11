import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.decomposition import PCA

class Clustering:
    def __init__(self, df, algorithm:str=None, linkage_method:str=None, features:list=None):
        """
        Initialize the clustering class with the stock dataset.
        """
        self.df = df.copy()
        self.scaled_features = None
        self.df_features = None
        self.df_pca = None
        assert algorithm in ["hierarchical", "DBSCAN"], 'Pass one algorithm between "hierarchical", "DBSCAN"'
        self.algorithm = algorithm
        self.linkage_method = linkage_method
        if features is not None:
            self.features = features
        else: 
            self.features = ['mc', 'mc_retail', 'normalized_holders','prc_adj', 'shrcd', 'rank_mkt', 'rank_ret', 'rank_distance', 'rolling_volatility']

    def calculate_features(self, df):
        """Calculate engineered features from Robintrack data"""
        
        # Create a copy to avoid modifying the original
        data = self.df.copy()
        
        # 1. Basic transformations
        # Normalize holder counts by users to get percentage
        data['holder_percentage'] = data['holders'] / data['users'] * 100
        
        # Calculate log returns of holders (better for analysis than raw changes)
        data['log_holder_change'] = np.log(data['holders'] / data['holders'].shift(1))
        
        # 2. Create groupby object for ticker-level operations
        grouped = data.groupby('ticker')
        
        # Create empty dataframes for ticker-level metrics
        ticker_metrics = []
        
        # 3. Calculate features for each ticker
        for ticker, group in tqdm(grouped, desc="Processing tickers"):
            # Sort by date
            group = group.sort_values('date')
            
            # Skip if not enough data
            if len(group) < 30:
                continue
                
            # Calculate time-based features
            group['holders_7d_pct_change'] = group['holders'].pct_change(periods=7)
            group['holders_30d_pct_change'] = group['holders'].pct_change(periods=30)
            
            # Volatility metrics
            group['holders_7d_std'] = group['holders'].rolling(7).std()
            group['holders_30d_std'] = group['holders'].rolling(30).std()
            group['holders_7d_cv'] = group['holders_7d_std'] / group['holders'].rolling(7).mean()
            
            # Momentum indicators
            group['holders_7d_ma'] = group['holders'].rolling(7).mean()
            group['holders_30d_ma'] = group['holders'].rolling(30).mean()
            group['ma_cross_signal'] = np.where(group['holders_7d_ma'] > group['holders_30d_ma'], 1, -1)
            
            # Acceleration (second derivative)
            group['holder_change_acceleration'] = group['daily_change_holders'].diff()
            
            # Relationship to price
            group['price_holder_corr_30d'] = group['holders'].rolling(30).corr(group['prc'])
            
            # Calculate retail concentration
            group['retail_concentration'] = group['mc_retail'] / group['mc'] * 100
            
            # Identify trend strength using linear regression slope
            if len(group) >= 30:
                for window in [30, 90]:
                    if len(group) >= window:
                        group[f'trend_strength_{window}d'] = group['holders'].rolling(window=window).apply(
                            lambda x: stats.linregress(np.arange(len(x)), x)[0] / np.mean(x), raw=True
                        )
            
            # Relative strength compared to overall market
            group = group.dropna(subset=['holders'])
            
            # Add back to the list
            ticker_metrics.append(group)
        
        # Combine all ticker data
        result = pd.concat(ticker_metrics)
        
        # Fill NaN values for calculated columns
        numeric_cols = result.select_dtypes(include=['float64']).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        return result 


    def preprocess_data(self, build_features:bool=False):
        """
        Perform feature engineering and prepare data for clustering.
        """
        if build_features:
            # Build features
            self.df["rolling_volatility"] = self.df["prc_adj"].pct_change().rolling(window=10).std()
            self.df["normalized_holders"] = self.df["holders"] / self.df.groupby("date")["holders"].transform("sum")
            self.df["holders_adj"] = self.df["holders"] / self.df.groupby("date")["holders"].transform("sum")
            self.df["holders_adj_change"] = self.df["holders_adj"].pct_change()

        # Select relevant features
        print(self.df.dtypes)
        #self.df_features = self.df.groupby("ticker")[self.features].mean()
        self.df_features = self.df
        self.df_features = self.df_features.dropna()

        # Build Time features

        # Standardize features
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.df_features)

    def apply_dbscan(self, eps=1.5, min_samples=5):
        """
        Apply DBSCAN clustering to the dataset.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.df_features["cluster"] = dbscan.fit_predict(self.scaled_features)

        # Append tickers to the dataframe
        #ticker_mapper = self.df["ticker"].to_dict()
        #self.df_features["ticker"] = self.df_features.index.map(ticker_mapper)

    def apply_hierarchical_clustering(self):
        linkage_matrix = linkage(self.scaled_features, method=self.linkage_method)
        return linkage_matrix


    def apply_pca(self, n_components=2):
        """
        Apply PCA for dimensionality reduction.
        """
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(self.scaled_features)
        self.df_pca = pd.DataFrame(pca_features, columns=["PCA1", "PCA2"])
        if self.algorithm == "DBSCAN":
            self.df_pca["cluster"] = self.df_features["cluster"].values
                        

    def plot_clusters(self):
        """
        Plot the clusters in PCA-reduced space.
        """

    
        plt.figure(figsize=(20, 7))
        if self.algorithm == "hierarchical":
            linked = self.apply_hierarchical_clustering()
            dendrogram(linked)
            plt.title("Dendrogram for Hierarchical Clustering")
            plt.xlabel("Tickers")
            plt.ylabel("Distance")
        else:
            plt.show()
            sns.set_style("whitegrid")
            sns.scatterplot(data=self.df_pca, x="PCA1", y="PCA2", hue="cluster", palette="tab10", alpha=0.7)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title(f"{self.algorithm} Clustering with PCA Reduction")
            plt.legend(title="Cluster")
            plt.grid(True, linestyle="--", alpha=0.6)
        

    def run_pipeline(self, eps=1.5, min_samples=5, build_features:bool=False):
        """
        Run the entire clustering pipeline: preprocessing, DBSCAN, PCA, and plotting.
        """
        self.preprocess_data(build_features)
        if self.algorithm == "DBSCAN":
            self.apply_dbscan(eps, min_samples)
        
        elif self.algorithm == "hierarchical":
            self.apply_hierarchical_clustering()
        
        self.apply_pca()
        self.plot_clusters()

