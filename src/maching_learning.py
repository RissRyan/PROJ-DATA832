import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, kmeans_plusplus


def pca_on_dataframe (dataframe):
    """
    Appliquer une ACP sur un dataframe
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à traiter

        Sorties :
            pca_data (list) : le résultat brut de l'ACP
            pca_data_country_index (pandas.Dataframe) : le résultat de l'ACP avec les labels
    """

    # STANDARDISATION
    scaler = StandardScaler()
    cols = [col for col in dataframe.columns if col != 'country']
    standardized_dataframe = dataframe.copy()
    # Standardiser les données des colonnes sélectionnées
    standardized_dataframe[cols] = scaler.fit_transform(standardized_dataframe[cols])

    # REDUCTION DIMENSION ACP
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_dataframe[cols])

    columns, country_index = ['PC1','PC2'], dataframe.country
    pca_data_country_index = pd.DataFrame(pca_data, columns = columns, index=country_index)

    return pca_data, pca_data_country_index


def kmeans_clustering (dataframe, cluster_number):
    """
    Appliquer la méthode kmeans sur un jeu de données
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à traiter
            cluster_number (int) : le nombre de clusters

        Aucune sortie
    """

    pca_data_country_index = pca_on_dataframe(dataframe)[1]

    # CLUSTERING KMEANS
    kmeans = KMeans(n_clusters=cluster_number,
                    random_state=0,
                    n_init=cluster_number).fit(pca_data_country_index)


    # AJOUT DES LABELS DU KMEANS
    kmean_df = pca_data_country_index.copy()
    kmean_df['cluster'] = kmeans.labels_

    # VOIR REPARTITION DES CLUSTERS
    sbn.swarmplot(data=kmean_df.cluster)
    plt.xlabel('Pays')
    plt.ylabel('Numéro de cluster')
    plt.title('Répartition des pays selon les clusters\nKmeans')
    plt.show()

    # SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS
    sbn.scatterplot(data=kmean_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
    plt.title('Répartition des pays\nKmeans')
    plt.show()

    # AFFICHER LA LISTE DES PAYS DANS CHAQUE CLUSTER
    for i in range (cluster_number):
        current_cluster_filter = kmean_df["cluster"].isin([i])
        current_cluster = kmean_df[current_cluster_filter].index
        print(current_cluster)



def kmeans_plus_plus_clustering (dataframe, cluster_number) :
    """
    Appliquer la méthode kmeans sur un jeu de donnée avec une initialisation optimisée
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à traiter
            cluster_number (int) : le nombre de clusters

        Aucune sortie
    """

    pca_data, pca_data_country_index = pca_on_dataframe(dataframe)


    # CALCUL DE L'INITIALISATION AVEC KMEANS++
    centers_init = kmeans_plusplus(pca_data, n_clusters=cluster_number, random_state=0)[0]

    # CLUSTERING KMEANS
    kmeans = KMeans(n_clusters=cluster_number,
                    init=centers_init,
                    n_init=1).fit(pca_data_country_index)


    # AJOUT DES LABELS DU KMEANS
    kmean_df = pca_data_country_index.copy()
    kmean_df['cluster'] = kmeans.labels_

    # VOIR REPARTITION DES CLUSTERS
    sbn.swarmplot(data=kmean_df.cluster)
    plt.xlabel('Pays')
    plt.ylabel('Numéro de cluster')
    plt.title('Répartition des pays selon les clusters\nKmeans++')
    plt.show()

    # SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS
    sbn.scatterplot(data=kmean_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
    plt.title('Répartition des pays\nKmeans++')
    plt.show()

    # AFFICHER LA LISTE DES PAYS DANS CHAQUE CLUSTER
    clusters = []
    for i in range (cluster_number):
        current_cluster_filter = kmean_df["cluster"].isin([i])
        current_cluster = kmean_df[current_cluster_filter].index
        clusters.append(list(current_cluster))
        print(list(current_cluster))
    return clusters


def bdscan_clustering (dataframe, eps, min_samples):
    """
    Appliquer la méthode DBSCAN sur un jeu de données
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à traiter
            eps (float) : l'espacement necessaire entre les points
            min_samples (int) : le nombre de voisins necessaire

        Aucune sortie
    """

    pca_data_country_index = pca_on_dataframe(dataframe)[1]

    bdscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pca_data_country_index)

    dbscan_df = pca_data_country_index.copy()
    dbscan_df['cluster'] = bdscan.labels_

    # SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS
    sbn.scatterplot(data=dbscan_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
    plt.title('Répartition des pays\nBDSCAN')
    plt.show()
    plt.show()
