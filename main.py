import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

 
countryDF = pd.read_csv('data/Country-data.csv')

#print(countryDF.head())
#print(countryDF.info()) # Toutes les features sont des réels sauf "country"

# Calculer la moyenne de chaque feature
means = countryDF.mean(numeric_only=True)
print("Moyenne de chaque feature :")
print(means)

# Calculer la variance de chaque feature
variances = countryDF.var(numeric_only=True)
print("Variance de chaque feature :")
print(variances)

# Calculer la matrice de corrélation
corr_matrix = countryDF.corr(numeric_only=True)

# Afficher la matrice de corrélation
print("Matrice de corrélation :")
print(corr_matrix)

"""
# Récupérer toutes les colonnes sauf la colonne "country"
cols = [col for col in countryDF.columns if col != 'country']

# Créer des graphiques catplot() pour chaque variable
for col in cols:
    sbn.catplot(x='country', y=col, kind='box', data=countryDF)
# Afficher les graphiques
plt.show()"""


def topCountriesByFeature(df, feature, n=10, ascendingParam=False):

    # Trier le DataFrame en fonction de la feature choisie
    sortedDF = df.sort_values(by=feature, ascending=ascendingParam)

    topCountries = sortedDF.head(n)

    # Créer le graphique
    sbn.barplot(x='country', y=feature, data=topCountries)

    plt.xlabel('Pays')
    plt.ylabel(feature)
    plt.title(f"Top {n} pays avec les valeurs les plus {'faibles' if ascendingParam else 'élevées'} pour '{feature}'")

    plt.show()

topCountriesByFeature(countryDF, 'health', 10)