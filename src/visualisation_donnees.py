import seaborn as sbn
import matplotlib.pyplot as plt

# print(countryDF.head())
# print(countryDF.info())
# Toutes les features sont des réels sauf "country"

def afficher_moyennes_colonnes (dataframe):
    """
    Affiche dans la console la moyenne de chaque colonne du dataframe.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser

        Sorties :
            bool : True
    """
    # Calculer la moyenne de chaque feature
    means = dataframe.mean(numeric_only=True)
    print("Moyenne de chaque feature :")
    print(means)

def afficher_variances_colonnes (dataframe):
    """
    Affiche dans la console la variance de chaque colonne du dataframe.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser

        Sorties :
            bool : True
    """
    # Calculer la variance de chaque feature
    variances = dataframe.var(numeric_only=True)
    print("Variance de chaque feature :")
    print(variances)


def afficher_matrice_covariance (dataframe):
    """
    Affiche dans la console la matricce de covariance du dataframe.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser

        Sorties :
            bool : True
    """
    # Calculer la matrice de corrélation
    corr_matrix = dataframe.corr(numeric_only=True)

    # Afficher la matrice de corrélation
    print("Matrice de corrélation :")
    print(corr_matrix)


def distribution_variable(dataframe, column):
    #sbn.catplot(x='country', y=column, kind='box', data=dataframe)
    sbn.displot(dataframe, x=column)
    # Afficher les graphiques
    plt.show()



def top_countries_by_feature(dataframe, feature, number_element=10, ascending_param=False):
    """
    Affiche un histogramme des pays ayant les valeurs les plus extrêmes selon une colonne.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser
            feature (string) : le nom de la colonne à visualiser
            number_element (int) : nombre des elements à afficher (default = 10)
            ascending_param (bool) : affichage par ordre croissant ou non (default = False)

        Sorties :
            bool : True
    """

    # Trier le DataFrame en fonction de la feature choisie
    sorted_dataframe = dataframe.sort_values(by=feature, ascending=ascending_param)

    top_countries = sorted_dataframe.head(number_element)

    # Créer le graphique
    sbn.barplot(x='country', y=feature, data=top_countries)

    plt.xlabel('Pays')
    plt.ylabel(feature)
    ordre = 'faibles' if ascending_param else 'élevées'
    plt.title(f"Top {number_element} pays avec les valeurs les plus {ordre} pour '{feature}'")

    plt.show()
    return True
