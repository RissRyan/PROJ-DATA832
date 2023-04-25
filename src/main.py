import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import visualisation_donnees as vd


### VISUALISATIONS

country_dataframe = pd.read_csv('data/Country-data.csv')

# vd.afficher_moyennes_colonnes(country_dataframe)
# vd.afficher_variances_colonnes(country_dataframe)
# vd.afficher_matrice_covariance(country_dataframe)
# vd.afficher_graph_colonnes(country_dataframe, 'health')
"""
cols = [col for col in country_dataframe.columns if col != 'country']
for col in cols:
    vd.distribution_variable(country_dataframe, col)
    vd.top_countries_by_feature(country_dataframe, col, ascending_param=True)"""


### CLUSTERING AVEC KMEANS


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

print(kmean_df)

# VOIR REPARTITION DES CLUSTERS

sbn.swarmplot(data=kmean_df.cluster)
plt.show()

# SCATTER PLOTS DES POINTS ET COULEURS AVEC LES CLUSTERS

sbn.scatterplot(data=kmean_df, x='PC1', y='PC2', hue='cluster', palette="tab10")
plt.show()

