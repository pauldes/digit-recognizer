import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
version='min-'
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

# Sampling %
SAMPLING_RATE = 0.1
print("Sampling training data - ",SAMPLING_RATE)
sample = np.random.randint(csvtrain.shape[0], size=int(csvtrain.shape[0]*SAMPLING_RATE))
data = pixels[sample,:]
target = labels[sample]

# PCA!
'''
std_scale = preprocessing.StandardScaler().fit(data)
data = std_scale.transform(data)
#target = std_scale.transform(target)
data = pca.transform(data)
#target = pca.transform(target)
'''

############
# STEP 2
############

# Splitting 80%
print("Splitting training data")
xtrain, xtest, ytrain, ytest = train_test_split(data,target,train_size=0.8,test_size=0.2)

############
# STEP 2-BISBIS
############

errors = []
for k in range(1,11):
    print("Fitting with k=",k)
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(xtrain, ytrain)
    errors.append([k , 100*(1-knn.score(xtest, ytest))])
errors = np.array(errors)

MIN_ERR_K = int(errors[:,0][np.argmin(errors[:,1])])
print("Minimum error with k=",MIN_ERR_K)
plt.plot(errors[:,0],errors[:,1])
plt.scatter(MIN_ERR_K,np.min(errors[:,1]),c='red')
plt.ylabel('Error rate (%)')
plt.title('Error rate with neighbours number')
plt.show()



############
# STEP 3
############

print(xtrain.shape)

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
knn = neighbors.KNeighborsClassifier(n_neighbors=MIN_ERR_K)
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
np.savetxt ('./csv/submission.csv', results, delimiter=",",fmt="%i",header="ImageId,Label",comment="")
