import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('./csv/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

test_images[test_images>0]=1
train_images[train_images>0]=1

clf = svm.SVC(kernel='rbf', random_state=0, gamma=.01, C=1000)
clf.fit(train_images, train_labels.values.ravel())

test_data=pd.read_csv('./csv/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])

df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('./csv/res_svm_svc_binary.csv', header=True)