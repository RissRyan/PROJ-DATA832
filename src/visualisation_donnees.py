import seaborn as sbn
import matplotlib.pyplot as plt


def afficher_moyennes_colonnes (dataframe):
    """
    Affiche dans la console la moyenne de chaque colonne du dataframe.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser

        Aucune sortie
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

        Aucune sortie
    """

    # Calculer la variance de chaque feature
    variances = dataframe.var(numeric_only=True)
    print("Variance de chaque feature :")
    print(variances)


def afficher_matrice_covariance (dataframe):
    """
    Affiche dans la console la matrice de covariance du dataframe.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser

        Aucune sortie
    """

    # Calculer la matrice de corrélation
    corr_matrix = dataframe.corr(numeric_only=True)

    # Afficher la matrice de corrélation
    print("Matrice de corrélation :")
    print(corr_matrix)


def distribution_variable(dataframe, column):
    """
    Affiche dans la distribution d'une variable d'un dataframe
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe qui comporte la donnée
            column (str) : le nom de la colonne à visualiser

        Aucune sortie
    """

    sbn.displot(dataframe, x=column)
    plt.show()


def top_countries_by_feature(dataframe, feature, number_element=10, ascending_param=False):
    """
    Affiche un histogramme des pays ayant les valeurs les plus extrêmes selon une colonne.
        Paramètres :
            dataframe (pandas.Dataframe) : le dataframe à visualiser
            feature (string) : le nom de la colonne à visualiser
            number_element (int) : nombre des elements à afficher (default = 10)
            ascending_param (bool) : affichage par ordre croissant ou non (default = False)

        Aucune sortie
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
