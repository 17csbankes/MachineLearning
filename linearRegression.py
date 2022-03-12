from sklearn import linear_model
import pandas as pd
import numpy as np
import math

# Load in data
dataFrame = pd.read_csv('/Users/christopherbankes/Desktop/Chris\'s Stuff/MachineLearning/prices.csv')

# Need to examine data and make sure it's normalized and all filled:


# We will use the median bedrooms and floor bc no half/partial bathrooms:
# Find any errors (empty/non-numeric values) in the data in convert them to numeric values:
dataFrame['Bedrooms'] = pd.to_numeric(dataFrame['Bedrooms'], errors='coerce')
print(dataFrame)
# Convert our data from strings to floats:
dataFrame = dataFrame.astype(float)
# Fill all NaN values with a value that makes sense for the data
# in this case we used the median of all bedrooms
median_bedrooms = math.floor(dataFrame.Bedrooms.median())
dataFrame.Bedrooms = dataFrame.Bedrooms.fillna(median_bedrooms)

# Create a linear regression model and populate it with our data:
model = linear_model.LinearRegression()
model.fit(dataFrame[['Area','Bedrooms','Age']], dataFrame.Price)

# Data for Line of Best fit:
# price = M1*X1 + M2*X2 + M3*X3 + b
# where:
# X1, X2, X3 are our independent variables (features - area, bedrooms, age)
# M1, M2, M3 are the coefficients for the Line of Best fit, found by the Linear Regression Model
# b is the y-intercept of the line of best fit

mCoefArray = model.coef_
bIntercept = model.intercept_
sqFt = 6000
bedrooms = 3
age = 79
print("Prediction of a house with {0} sqFt, {1} bedrooms, and {2} yrs old: ${3}".format(sqFt, bedrooms, age, model.predict([[sqFt,bedrooms,age]])[0]))

# Examples of accessing arrays from a dataFrame object:
# Both return the same type, same data
"""
print(dataFrame.Bedrooms)
print(type(dataFrame.Bedrooms), "\n")

print(dataFrame['Bedrooms'])
print(type(dataFrame['Bedrooms']))"""