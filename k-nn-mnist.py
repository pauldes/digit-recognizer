#####################################
# K-NearestNeigbourh classification
# For the MNIST dataset
# ###################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import neighbors

mnist = fetch_mldata('MNIST original')

#Grand k -> Lissage, moins sensible au bruit
#Petit k -> Structures plus fines
#Regarder les k voisins les plus proches de ce point et regarder quelle classe constitue la majorité de ces points afin d'en en déduire la classe du nouveau point
#k->infini: Bayes

####
#k n'est pas un paramètre mais un hyperparamètre, c'est à dire que contrairement aux paramètres classiques, il ne va pas pouvoir être appris automatiquement par l'algorithme à partir des données d'entraînement.
#Les hyperparamètres permettent de caractériser le modèle (e.g. complexité, rapidité de convergence etc).
#Ce ne sont pas les données d'apprentissage qui vont permettre de trouver ces paramètres (en l'occurence ici le nombre de voisins k) mais bien à nous de l'optimiser, à l'aide du jeu de données test.
#####

# Le dataset principal qui contient toutes les images
print(mnist.data.shape)
#784 features: 28pixels*28pixels*1color(grayscales)
#If in colors RGB: 28*28*3
# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print(mnist.target.shape)
# A number between 0 and 9

# Sampling 20%
sample = np.random.randint(70000, size=int(7000*0.2))
data = mnist.data[sample,:]
target = mnist.target[sample]

# Splitting 80%
xtrain, xtest, ytrain, ytest = train_test_split(data,target,train_size=0.8,test_size=0.2)

'''
# 3-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
# ytrain (labels) has 10 different values
# => there will be 10 classes (clusters)
# Test
error = 1 - knn.score(xtest, ytest)
print('Erreur: ',error)
'''

# Finding best K hyper-parameter

errors = []
for k in range(1,20):
    print("Fitting with k=",k)
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(xtrain, ytrain)
    errors.append([k , 100*(1-knn.score(xtest, ytest))])
errors = np.array(errors)

minerrk = errors[:,0][np.argmin(errors[:,1])]
print("Minimum error with k=",minerrk)

plt.plot(errors[:,0],errors[:,1])
plt.axhline(np.min(errors[:,1]),c='red')
plt.axvline(minerrk,c='red')
plt.xlabel('K-NearestNeigbourh')
plt.ylabel('Error rate (%)')
plt.title('Error rate with neighbors number')
plt.show()


# Fitting with optimal K and full training set
data = mnist.data[1:-1,:]
target = mnist.target[1:-1]
xtrain, xtest, ytrain, ytest = train_test_split(data,target,train_size=0.8,test_size=0.2)
knn = neighbors.KNeighborsClassifier(n_neighbors=minerrk)
print('Fitting with the full training set ... ',xtrain.shape)
knn.fit(xtrain, ytrain)
print('Testing with the full testing set ... ',xtest.shape)
error = 1-knn.score(xtest, ytest)
print('Final error: ',error)
