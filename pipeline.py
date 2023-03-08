import pandas as pd
import warnings 
warnings.filterwarnings("ignore")

from algorithms.clustering.kmeans import KMeans_Clustering
from algorithms.associate_rule_mining.apriori import Apriori

import utils.clustering as clustering_utils
import utils.forecasting as forecasting_utils

ecommerce_df = pd.read_csv("./data/e-commerce-data.csv", encoding= 'unicode_escape')

kmeans_model = KMeans_Clustering()
clustering_df = clustering_utils.preprocess(ecommerce_df)
customer_clusters_kmeans = kmeans_model.k_means_fit(clustering_df)

apriori_model = Apriori()
associate_rules = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)

forecasting_df = forecasting_utils.preprocess(ecommerce_df)
forecasting_utils.visualize(forecasting_df)