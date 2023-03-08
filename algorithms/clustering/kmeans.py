from sklearn.cluster import KMeans

class KMeans_Clustering:

    def __init__(self) -> None:
        pass

    def k_means_fit(self, ecommerce_df):
        kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(ecommerce_df[["Amount","Frequency"]])
        ecommerce_df['Cluster'] = y_kmeans
        return ecommerce_df