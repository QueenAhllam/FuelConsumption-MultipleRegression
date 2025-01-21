import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures

data =pd.read_csv("FuelConsumption.csv")

X_1 = data.iloc[:,4:5].values
X_2 = data.iloc[:,5:6].values
X_3 = data.iloc[:,9:10].values

X= np.concatenate((X_1,X_2,X_3), axis =1) 
y = data.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train) #model is trained
print("theta0", model.intercept_)
print("theta1", model.coef_)

y_hat = model.predict(X_test)

print("MSE: ", mean_squared_error(y_test, y_hat))
print("r2: ", r2_score(y_test, y_hat))

This is a Level 2 project focusing on Multiple Linear Regression.Written by Queen Ahllam 👑.




