import pygal
from pycountry_convert import country_name_to_country_alpha2

def generation_carte (clusters) :
    """
    Genere une carte pour representer les clusters.
        Paramètres :
            clusters (liste) : la liste des clusters de pays

        Aucune sortie
    """

    dicts = [{}, {}, {}]
    print("\nListe des pays non affichés : ")
    for i in enumerate (clusters) :
        for j in range (len(clusters[i[0]])) :
            try :
                code = country_name_to_country_alpha2(clusters[i[0]][j])
                dicts[i[0]][code.lower()] = i[0]
            except :
                # Gestion de quelqeus exceptions
                if clusters[i[0]][j] == "Congo, Rep." :
                    dicts[i[0]]['cg'] = i[0]
                elif clusters[i[0]][j] == "Congo, Dem. Rep." :
                    dicts[i[0]]['cd'] = i[0]
                elif clusters[i[0]][j] == "Cote d'Ivoire" :
                    dicts[i[0]]['ci'] = i[0]
                else :
                    print("   ", clusters[i[0]][j])


    worldmap = pygal.maps.world.World()
    worldmap.title = "Clustering des pays selon leurs besoins en soutien financier"
    worldmap.add("Pas besoin d'aide", dicts[1], color = 'green')
    worldmap.add("Peut être besoin d'aide", dicts[0], color = 'yellow')
    worldmap.add("Besoin d'aide", dicts[2], color = 'red')
    worldmap.render_to_file('map.svg')
