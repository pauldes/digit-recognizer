import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

# Get data
print("Getting training data")
csvtrain = np.genfromtxt ('./csv/train.csv', delimiter=",")
# Remove title line
csvtrain = csvtrain[1:,:]
# Select pixels and labels
labels = csvtrain[:,0]
pixels = csvtrain[:,1:]


# Re-process data
X = pixels
Y = labels
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
pca = decomposition.PCA(n_components=200) #xNinetyNine
pca.fit(X_scaled)
X_projected = pca.transform(X_scaled)
print("Training K-NN on full dataset")
knn = neighbors.KNeighborsClassifier(n_neighbors=7) #MIN_ERR_K
knn.fit(X_projected, Y)

# Get data
print("Getting testing data")
csvpred = np.genfromtxt ('./csv/test.csv', delimiter=",")
# Remove title line
Xpred = csvpred[1:,:]
print("Scaling and projecting")
# We apply the same StandardScaler
Xpred_scaled = std_scale.transform(Xpred)
# We apply the same PCA
Xpred_projected = pca.transform(Xpred_scaled)

# Predict
print("Predicting")
ypred = knn.predict(Xpred_projected).astype(int)
# Save in file with index
print("Indexing")
indexes = np.arange(1,ypred.shape[0]+1)
results = np.vstack([indexes,ypred])
print("Transposing")
results = results.transpose()
print("Saving")
np.savetxt ('./csv/submission.csv', results, delimiter=",",fmt="%i",header="ImageId,Label")

