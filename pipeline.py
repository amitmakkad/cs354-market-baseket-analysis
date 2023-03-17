import warnings 
warnings.filterwarnings("ignore")

import sys
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np

from algorithms.clustering import hierarchical, kmeans , dbscan
from algorithms.associate_rule_mining.apriori import Apriori

import utils.clustering as clustering_utils
import utils.forecasting as forecasting_utils



ecommerce_df = pd.read_csv("./data/e-commerce-data.csv", encoding= 'unicode_escape')

clustering_df   = clustering_utils.preprocess(ecommerce_df)
clustering_data = clustering_utils.get_data(clustering_df)

kmeans_model = kmeans.KMeans(k=6, max_iters=10)
kmeans_model.fit(clustering_data)
kmeans_labels = kmeans_model.predict(clustering_data)

dbscan_model = dbscan.DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan_model.fit(clustering_data)

hierarchical_model = hierarchical.HierarchicalClustering(n_clusters=2)
hierarchical_model.fit(clustering_data)
hierarchial_labels = hierarchical_model.predict(clustering_data)

customer_clusters_kmeans, customer_clusters_dbscan, customer_clusters_hierarchial = ecommerce_df, ecommerce_df, ecommerce_df
customer_clusters_kmeans['Cluster'] = kmeans_labels
customer_clusters_dbscan['Cluster'] = dbscan_labels
customer_clusters_hierarchial['Cluster'] = hierarchial_labels

customer_clusters = get_best_clusters(customer_clusters_kmeans, customer_clusters_dbscan, customer_clusters_hierarchial)

apriori_model = Apriori()
associate_rules_apriori = apriori_model.get_associate_rules(ecommerce_df, customer_clusters)

fp_growth_model = Apriori()
associate_rules_fp_growth = fp_growth_model.get_associate_rules(ecommerce_df, customer_clusters)

associate_rules = get_best_associate_rules(associate_rules_apriori, associate_rules_fp_growth)


forecasting_df = forecasting_utils.preprocess(ecommerce_df)
forecasting_utils.visualize(forecasting_df)