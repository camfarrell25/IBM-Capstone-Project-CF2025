import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline 
import requests


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
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()

#Plot the features 
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

#Plot each of the features against emmissions

features = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']

for feature in features:
    plt.figure()
    plt.scatter(cdf[feature], cdf.CO2EMISSIONS, color = 'blue')
    plt.xlabel(f"{feature}")
    plt.ylabel('Emission')
    plt.title(f'{feature} against CO2EMISSIONS')
    plt.show() 

#Engine Size - Positive Linear Relationship
#Cylinder - Positive 
#Fuel Consumption - Postive 


#Creating a test and train dataset
#Split 
msk = np.random.rand(len(df)) < 0.8
train  = cdf[msk]
test = cdf[~msk]

#Train data distribution 
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')
plt.show()

#Multiple Regression Model 
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('Coefficients: ', regr.coef_)

#Prediction 
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("MSE : %.2f" %np.mean((y_hat - y)**2))
#Explained Variance Score - 1 is perfect prediction
print('Variance score: %.2f' %regr.score(x,y))


#Multiple Regression Model 
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('Coefficients: ', regr.coef_)

#Prediction 
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("MSE : %.2f" %np.mean((y_hat - y)**2))
#Explained Variance Score - 1 is perfect prediction
print('Variance score: %.2f' %regr.score(x,y))