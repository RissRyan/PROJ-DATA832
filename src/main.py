import pandas as pd
import visualisation_donnees as vd

country_dataframe = pd.read_csv('data/Country-data.csv')

# vd.afficher_moyennes_colonnes(country_dataframe)
# vd.afficher_variances_colonnes(country_dataframe)
# vd.afficher_matrice_covariance(country_dataframe)
# vd.afficher_graph_colonnes(country_dataframe, 'health')

cols = [col for col in country_dataframe.columns if col != 'country']
for col in cols:
    vd.top_countries_by_feature(country_dataframe, col, ascending_param=True)
