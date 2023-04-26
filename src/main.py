import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, kmeans_plusplus
from sklearn.datasets import make_blobs
import visualisation_donnees as vd


### VISUALISATIONS

country_dataframe = pd.read_csv('data/Country-data.csv')

# vd.afficher_moyennes_colonnes(country_dataframe)
# vd.afficher_variances_colonnes(country_dataframe)
# vd.afficher_matrice_covariance(country_dataframe)
# vd.afficher_graph_colonnes(country_dataframe, 'health')

# cols = [col for col in country_dataframe.columns if col != 'country']
# for col in cols:
#     vd.distribution_variable(country_dataframe, col)
#     vd.top_countries_by_feature(country_dataframe, col, ascending_param=True)




### CLUSTERING AVEC KMEANS

def kmeans () :

    # STANDARDISATION
    scaler = StandardScaler()
    cols = [col for col in country_dataframe.columns if col != 'country']
    standardized_country_dataframe = country_dataframe.copy()
    # Standardiser les données des colonnes sélectionnées
    standardized_country_dataframe[cols] = scaler.fit_transform(standardized_country_dataframe[cols])

    # REDUC DIM PCA

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_country_dataframe[cols])

    pca_data_countryIndex = pd.DataFrame(pca_data, columns = ['PC1','PC2'], index=country_dataframe.country)


    # CLUSTERING KMEANS
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pca_data_countryIndex)


    # AJOUT DES LABELS DU KMEANS
    kmean_df = pca_data_countryIndex.copy()
    kmean_df['cluster'] = kmeans.labels_

    # print(kmean_df)

    # VOIR REPARTITION DES CLUSTERS

    sbn.swarmplot(data=kmean_df.cluster)
    plt.show()

    # SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS

    sbn.scatterplot(data=kmean_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
    plt.show()


    cluster0_filter = kmean_df["cluster"].isin([0])
    cluster0 = kmean_df[cluster0_filter].index
    print(cluster0)

    cluster1_filter = kmean_df["cluster"].isin([1])
    cluster1 = kmean_df[cluster1_filter].index
    print(cluster1)

    cluster2_filter = kmean_df["cluster"].isin([2])
    cluster2 = kmean_df[cluster2_filter].index
    print(cluster2)


def BDSCAN ():
    """
    Essai non concluant
    """
    scaler = StandardScaler()
    cols = [col for col in country_dataframe.columns if col != 'country']
    standardized_country_dataframe = country_dataframe.copy()
    # Standardiser les données des colonnes sélectionnées
    standardized_country_dataframe[cols] = scaler.fit_transform(standardized_country_dataframe[cols])

    # REDUC DIM PCA

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_country_dataframe[cols])

    pca_data_countryIndex = pd.DataFrame(pca_data, columns = ['PC1','PC2'], index=country_dataframe.country)

    bdscan = DBSCAN(eps=0.5, min_samples=3).fit(pca_data_countryIndex)


    # AJOUT DES LABELS DU KMEANS
    dbscan_df = pca_data_countryIndex.copy()
    dbscan_df['cluster'] = bdscan.labels_

    # SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS

    sbn.scatterplot(data=dbscan_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
    plt.show()



def kmeans_plus_plus () :
    """
    Attention : non fonctionelle
    """

    # STANDARDISATION
    scaler = StandardScaler()
    cols = [col for col in country_dataframe.columns if col != 'country']
    standardized_country_dataframe = country_dataframe.copy()
    # Standardiser les données des colonnes sélectionnées
    standardized_country_dataframe[cols] = scaler.fit_transform(standardized_country_dataframe[cols])

    # REDUC DIM PCA

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_country_dataframe[cols])

    pca_data_countryIndex = pd.DataFrame(pca_data, columns = ['PC1','PC2'], index=country_dataframe.country)


    # Generate sample data
    n_samples = 4000
    n_components = 3

    pca_data_countryIndex, y_true = make_blobs(
        n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
    )
    pca_data_countryIndex = pca_data_countryIndex[:, ::-1]

    # Calculate seeds from k-means++
    centers_init, indices = kmeans_plusplus(pca_data_countryIndex, n_clusters=3, random_state=0)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    for k, col in enumerate(colors):
        cluster_data = y_true == k
        plt.scatter(pca_data_countryIndex[cluster_data, 0], pca_data_countryIndex[cluster_data, 1], c=col, marker=".", s=10)

    plt.scatter(centers_init[:, 0], centers_init[:, 1], c="b", s=50)
    plt.title("K-Means++ Initialization")
    plt.xticks([])
    plt.yticks([])
    plt.show()


kmeans_plus_plus()