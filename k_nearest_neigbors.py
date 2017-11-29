import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split

########################################################################
# Author: github.com/pauldes
# Paul. D @INSA Lyon
# K-nearest-neigbor algorithm for the Kaggle Digit Recognizer challenge
# MNIST dataset
# Strategy:
# 1. Take rnd sample set from labelled data set (20%)
# 2. Split the sample set ina training set and a testing set (80%/20%)
# 3. Train a model on the sample training set
# 4. Validate the model by testing it with the sample testing set
# 5. Train the model on the whole labelled set
# 6. Predict unlabelled data.
########################################################################
#version='min-'
version=''

############
# STEP 1
############

# Get data
print("Getting training data")
csvtrain = np.genfromtxt ('./csv/'+version+'train.csv', delimiter=",")
# Remove title line
csvtrain = csvtrain[1:,:]
# Select pixels and labels
labels = csvtrain[:,0]
pixels = csvtrain[:,1:]

# Sampling 20%
print("Sampling training data")
sample = np.random.randint(csvtrain.shape[0], size=int(csvtrain.shape[0]*0.2))
data = pixels[sample,:]
target = labels[sample]

############
# STEP 2
############

# Splitting 80%
print("Splitting training data")
xtrain, xtest, ytrain, ytest = train_test_split(data,target,train_size=0.8,test_size=0.2)

############
# STEP 3
############

# 3-NN
print("Training K-NN")
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
# ytrain (labels) has 10 different values
# => there will be 10 classes (clusters)

############
# STEP 4
############

error = 1 - knn.score(xtest, ytest)
print('Erreur: ',error)
#Todo: find optimal k
# k=5 scores 0.96857
# k=3 scores 0.96800

############
# STEP 5
############

# Model is validated
# Now train with full dataset
print("Training K-NN on full dataset")
xtrain = pixels[:,:]
ytrain = labels[:]
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

############
# STEP 6
############

# Get data
print("Getting testing data")
csvpred = np.genfromtxt ('./csv/'+version+'test.csv', delimiter=",")
# Remove title line
csvpred = csvpred[1:,:]
xpred = csvpred[:,:]
# Predict
print("Predicting")
ypred = knn.predict(xpred).astype(int)
# Save in file with index
print("Indexing")
indexes = np.arange(1,ypred.shape[0]+1)
results = np.vstack([indexes,ypred])
print("Transposing")
results = results.transpose()
print("Saving")
np.savetxt ('./csv/submission.csv', results, delimiter=",",fmt="%i",header="ImageId,Label")
