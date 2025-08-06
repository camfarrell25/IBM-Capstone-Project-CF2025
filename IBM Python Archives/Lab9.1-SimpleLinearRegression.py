import matplotlib.pyplot as plt
import pandas as pd
import pylab as  
import numpy as np
#import wget as wget
import requests
#%matplotlib inline

#!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
response = requests.get(url)

with open("FuelConsumption.csv", "wb") as file:
    file.write(response.content)

df = pd.read_csv("FuelConsumption.csv")
#Take a look at the dataset 
df.head()

#Data Exploration
df.describe()
#Select some feature to explore 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()

#Plot the features 
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

#Plot each of the features against emmissions

features = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']

for feature in features:
    plt.figure()
    plt.scatter(cdf[feature], cdf.CO2EMISSIONS, color = 'blue')
    plt.xlabel(f"{feature}")
    plt.ylabel('Emission')
    plt.title(f'{feature} against CO2EMISSIONS')
    plt.show() 


#Creating and Train Test Dataset 
#Create a mask to select ranom rows using np.random.ran() function 
msk = np.random.rand(len(df)) <0.8
print(msk)
train = cdf[msk]
test=cdf[~msk]

#Simple Regression Model 

#Train data distribution

 plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')
plt.show() 

#Modeling - using sklearn packate to model the data 
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
#The Coefficients 
print(f'Coefficients :', regr.coef_)
print(f'Intercept :', regr.intercept_)

#Plot outputs 
plt.figure(figsize=(8, 6))  # Create a new figure
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', label='Scatter')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r', label='Line')
plt.xlabel("Engine Size")
plt.ylabel('Emissions')
plt.legend()  # Show legend with labels
plt.show()

#Evaluation 
#Mean Absolute Error (MAE)
#Mean Squared Error (MSE)
#Root Mean Squared Error (RMSE)
#R-squared 

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )