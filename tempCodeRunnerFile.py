ap=0
# fp=0
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

# if kmeans==max(kmeans,dbscan,hierar):
#     print("best cluster by kmeans")
#     customer_cluster=customer_clusters_kmeans
# elif dbscan==max(kmeans,dbscan,hierar):
#     print("best cluster by dbscan")
#     # customer_cluster=customer_clusters_dbscan
# elif hierar==max(kmeans,dbscan,hierar):
#     print("best cluster by hierar")
#     # customer_cluster=customer_clusters_hierar


# # forecasting_df = forecasting_utils.preprocess(ecommerce_df)
# # forecasting_utils.visualize(forecasting_df)