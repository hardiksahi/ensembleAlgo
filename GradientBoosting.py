import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x_train = None
y_train = None

learnerCount =150


def readData(X,Y):
    x_train = pd.read_csv(X, header = None).as_matrix()
    y_train = pd.read_csv(Y, header = None).as_matrix()
    return x_train, y_train
    

x_train, y_train = readData('housing_X_train.csv','housing_y_train.csv' )
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
dimension = x_train.shape[1]
dataCount = x_train.shape[0]

print("Dimensions of training data X", x_train.shape)
print("Dimensions of training data Y", y_train.shape)

y_pred = np.full((dataCount,1), np.mean(y_train))

print(y_pred.shape, type(y_pred), y_pred)

learner = []
for k in range(learnerCount):
    nGradient = y_train - y_pred ## Only because I am using GB for regression and using least squares
    treeReg = DecisionTreeRegressor(max_depth=5)
    treeReg.fit(x_train,nGradient) #Fitting X_train and nGradient
    learner.append(treeReg)
    pred = treeReg.predict(x_train)
    pred = np.reshape(pred, (pred.shape[0],1))
    y_pred = y_pred + pred # This is using gradient descent model


x_test, y_test = readData('housing_X_test.csv', 'housing_y_test.csv')
x_test = scaler.fit_transform(x_test)
print("Shape of x_test",x_test.shape[0])
y_test_pred = np.zeros((x_test.shape[0],1))
rate = 1
for k in range(learnerCount):
    currLearn = learner[k]
    predict = (currLearn.predict(x_test)).reshape(x_test.shape[0],1)
    y_test_pred = np.add(y_test_pred,rate*predict)
    
residualTest = y_test-y_test_pred
print("residualTest", residualTest) 

