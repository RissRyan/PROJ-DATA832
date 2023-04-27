import pandas as pd
from sklearn.pipeline import Pipeline
import visualisation_donnees as vd
import maching_learning as ml


### IMPORTATION DES DONNEES
country_dataframe = pd.read_csv('data/Country-data.csv')


### VISUALISATIONS
vd.afficher_moyennes_colonnes(country_dataframe)
vd.afficher_variances_colonnes(country_dataframe)
vd.afficher_matrice_covariance(country_dataframe)

cols = [col for col in country_dataframe.columns if col != 'country']
for col in cols:
    vd.distribution_variable(country_dataframe, col)
    vd.top_countries_by_feature(country_dataframe, col, ascending_param=True)


### CLUSTERING
kmean_pipe = Pipeline(('PCA', ml.pca_on_dataframe(country_dataframe),
                       ('kmeans', ml.kmeans_clustering(country_dataframe, 3))))


kmean_plus_plus_pipe = Pipeline(('PCA', ml.pca_on_dataframe(country_dataframe),
                                ('kmeans++', ml.kmeans_plus_plus_clustering(country_dataframe, 3))))

dbscan_pipe = Pipeline(('PCA', ml.pca_on_dataframe(country_dataframe),
                       ('bdscan', ml.bdscan_clustering(country_dataframe, 0.45, 4))))

### ANALYSE
clusters = ml.kmeans_plus_plus_clustering(country_dataframe, 3)
vd.detailler_clusters(country_dataframe, clusters)
