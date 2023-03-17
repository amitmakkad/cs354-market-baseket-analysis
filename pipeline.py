import warnings 
warnings.filterwarnings("ignore")

import sys
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np

from algorithms.clustering import hierarchical, kmeans , dbscan
from algorithms.associate_rule_mining.apriori import Apriori
from algorithms.associate_rule_mining.fpgrowth import Fpgrowth

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

# apriori_model = Apriori()
# associate_rules_kmeans = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)
# fpgrowth_model = Fpgrowth()
# associate_rules_kmeans = fpgrowth_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)
# associate_rules_dbscan = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_dbscan)
# associate_rules_hierar = apriori_model.get_associate_rules(ecommerce_df, customer_clusters_kmeans)

# associate_rules_dbscan=associate_rules_kmeans.copy()
# associate_rules_hierar=associate_rules_kmeans.copy()

# associate_rules_kmeans.rename(columns={"lift": "lift1"}, inplace=True)
# associate_rules_kmeans.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

# associate_rules_dbscan.rename(columns={"lift": "lift2"}, inplace=True)
# associate_rules_dbscan.drop(['leverage', 'conviction','antecedent support','consequent support', 'support','confidence'], axis=1, inplace=True)

# associate_rules_hierar.rename(columns={"lift": "lift3"}, inplace=True)
# associate_rules_hierar.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence'], axis=1, inplace=True)

# print(associate_rules_kmeans)
# print(associate_rules_dbscan)
# print(associate_rules_hierar)
# df=pd.merge(pd.merge(associate_rules_kmeans,associate_rules_dbscan,on=['antecedents','consequents'],how="inner"),associate_rules_hierar,on=['antecedents','consequents'],how="inner")
# print(df)

kmeans=1
dbscan=0
hierar=0
# for index, row in df.iterrows():
#     l1=row['lift1']
#     l2=row['lift2']
#     l3=row['lift3']
#     if l1>l2 and l1>l3:
#         kmeans=kmeans+1
#     elif l2>l1 and l2>l3:
#         dbscan=dbscan+1
#     elif l3>l1 and l3>l2:
#         hierar=hierar+1

if kmeans==max(kmeans,dbscan,hierar):
    print("best cluster by kmeans")
    customer_cluster=customer_clusters_kmeans
elif dbscan==max(kmeans,dbscan,hierar):
    print("best cluster by dbscan")
    # customer_cluster=customer_clusters_dbscan
elif hierar==max(kmeans,dbscan,hierar):
    print("best cluster by hierar")
    # customer_cluster=customer_clusters_hierar

apriori_model = Apriori()
associate_rules_apriori = apriori_model.get_associate_rules(ecommerce_df, customer_clusters)

fp_growth_model = Apriori()
associate_rules_fp_growth = fp_growth_model.get_associate_rules(ecommerce_df, customer_clusters)

associate_rules = get_best_associate_rules(associate_rules_apriori, associate_rules_fp_growth)
associate_rules_apriori = apriori_model.get_associate_rules(ecommerce_df, customer_cluster)

associate_rules_apriori['zhang metric1'] = None

for i, row in associate_rules_apriori.iterrows():
    supp_x = row['antecedent support']
    supp_y = row['consequent support']
    supp_xy = row['support']
    max_supp = max(supp_x, supp_y)
    
    zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
    associate_rules_apriori.at[i, 'zhang metric1'] = zhang_metric

associate_rules_apriori.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)

fpgrowth_model = Fpgrowth()
associate_rules_fpgrowth = fpgrowth_model.get_associate_rules(ecommerce_df, customer_cluster)

associate_rules_fpgrowth['zhang metric2'] = None

for i, row in associate_rules_fpgrowth.iterrows():
    supp_x = row['antecedent support']
    supp_y = row['consequent support']
    supp_xy = row['support']
    max_supp = max(supp_x, supp_y)
    
    zhang_metric = (supp_xy - supp_x * supp_y) / (max_supp - supp_x * supp_y)
    associate_rules_fpgrowth.at[i, 'zhang metric2'] = zhang_metric

associate_rules_fpgrowth.rename(columns={"zhang metric": "zhang metric2"}, inplace=True)
associate_rules_fpgrowth.drop(['leverage', 'conviction','antecedent support','consequent support','support','confidence','lift'], axis=1, inplace=True)

print(associate_rules_apriori)
print(associate_rules_fpgrowth)

df=pd.merge(associate_rules_apriori,associate_rules_fpgrowth,on=['antecedents','consequents'],how="inner")
print(df)

ap=0
fp=0
for index, row in df.iterrows():
    z1=row['zhang metric1']
    z2=row['zhang metric2']
    if z1>z2:
        ap=ap+1
    else:
        fp=fp+1

if ap==max(ap,fp):
    print("best rules by apriori")
    rules=associate_rules_apriori
else:
    print("best rules by apriori")
    rules=associate_rules_fpgrowth



# forecasting_df = forecasting_utils.preprocess(ecommerce_df)
# forecasting_utils.visualize(forecasting_df)