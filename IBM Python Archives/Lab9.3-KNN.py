"""
K-nearest neighbor is a supervised learning algorithm. Where data is trained with data points corresponding to their classification. To predict the class of a given data point, it takes into account the classes of the K-nearst
data points and chooses the class in which the majority of the K nearest data points belong to as the predicted class. 
""" 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

"""
Dataset - Demographic data to predict membership. Where the custcat has four possible values - 1-Basic 2-E-Service, 3-Plus Service and 4-Total Service 
Objective to build classifier to predict the class of unknown cases 
"""

#Load Data 

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

#Data visualization and analysis 
#how many of each class is in our dataset
df['custcat'].value_counts()
#Visualize
fig1 = df.hist(column='income', bins=50)
print(fig1)

#Feature set - let's define feature sets: X
df.columns
#To use scikit learn, we have to convert the pandas DF to numpy array 
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed','employ', 'retire', 'gender', 'reside']].values
X[0:5]

y = df['custcat'].values
y[0:5]

#Normalize Data - data standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train Test Split 
#Out of sample accuracy is the % of correct predictions that the model makes on data that the model has not been trained on. Doing a train and tes on the same dataset will likely lead to low out of sample accuracy, as  a result of overfitting 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#Classification 
#KNN
#Import Library
from sklearn.neighbors import KNeighborsClassifier

#Training - let's start the algorith,m with k=4 for now: 
k=4
#Train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh

#Predicting - we can use the model to make predictions on the test set: 
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy evaluation - in a multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function - it calculates how closely the actual and predicted labels are matched in the test set. 
from sklearn import metrics 
print("Train Set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set accuracy:", metrics.accuracy_score(y_test, yhat))

#Train set accuracy is 0.5475
#Test set accuracy is 0.32

#Build the model with k=6 

#Training - let's start the algorith,m with k=4 for now: 
k=6
#Train model and predict
neigh6 = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh6

#Predicting - we can use the model to make predictions on the test set: 
yhat6 = neigh6.predict(X_test)
yhat6[0:5]

#Accuracy evaluation - in a multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function - it calculates how closely the actual and predicted labels are matched in the test set. 
from sklearn import metrics 
print("Train Set Accuracy:", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set accuracy:", metrics.accuracy_score(y_test, yhat6))

#What about other Ks? 
#How can we choose the right value for K? The general soluation is to reserve a part of your data for testing the accurancy of the model, then choose k=1 for modeling, and calculate the accuracy of prediction using all samples in your test set, and repeat the process and see which is the best

Ks= 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range (1, Ks):
    #train model and predict
    neigh = KNeighborsClassifier(n_neighbors= n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]= np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

#plot the model accuracy for the different number of neighbors

plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc-1*std_acc, mean_acc +1 *std_acc, alpha = 0.1)
plt.fill_between(range(1,Ks), mean_acc - 3*std_acc, mean_acc+3*std_acc, alpha = 0.1, color = 'green')
plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)