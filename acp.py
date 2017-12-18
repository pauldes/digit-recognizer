import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition

##################
# Importing data
##################

# charger les données
data = pd.read_csv('decathlon.txt', sep="\t")
# éliminer les colonnes que nous n'utiliserons pas
my_data = data.drop(['Points', 'Rank', 'Competition'], axis=1)
# transformer les données en array numpy
X = my_data.values

##################
# Centrer !!!
##################

std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

##################
# Compute the PCA
##################

pca = decomposition.PCA(n_components=2)
pca.fit(X_scaled)

#################
# Details
#################

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

#################
# Results
#################

# projeter X sur les composantes principales
X_projected = pca.transform(X_scaled)

# afficher chaque observation
plt.scatter(X_projected[:, 0], X_projected[:, 1],
    # colorer en utilisant la variable 'Rank'
    c=data.get('Rank'))

plt.xlim([-5.5, 5.5])
plt.ylim([-4, 4])
plt.colorbar()
plt.show()


######################
# How each variable contribute to the principal components
######################

pcs = pca.components_

for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
    # Afficher un segment de l'origine au point (x, y)
    plt.plot([0, x], [0, y], color='k')
    # Afficher le nom (data.columns[i]) de la performance
    plt.text(x, y, data.columns[i], fontsize=12)

# Afficher une ligne horizontale y=0
plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

# Afficher une ligne verticale x=0
plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
plt.xlim([-0.7, 0.7])
plt.ylim([-0.7, 0.7])
plt.show()
