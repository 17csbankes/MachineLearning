import tensorflow as tf
import numpy as np
import yfinance as yf
from tensorflow import keras
import datetime
import pandas as pd



def numDaysAgo(pastDates):
    indexArray = []
    
    today = datetime.datetime.now()
    for date in pastDates:
        difference = today - date
        difference_in_s = difference.total_seconds() 
        days  = difference.days                         # Build-in datetime function
        days  = divmod(difference_in_s, 86400)[0]       # Divide by seconds in a day (86400), get first value of tuple
        indexArray.append(days)

    indexArray.reverse()
    print(indexArray)
    return indexArray
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Generate a model using Stochastic gradient descent for the optimizer function (theta) and mean_squared error for the loss function
model.compile(optimizer='sgd', loss='mean_squared_error')


gme = yf.Ticker("GME")
gmeData = gme.history(period='1mo',interval='1d')
print(gmeData['High'])
standardizedDates = numDaysAgo(gmeData.index)
xData = np.array(standardizedDates, dtype=float)
yData = np.array(gmeData['High'], dtype=float)

#model.fit(xData, yData, epochs=500)

#print(model.predict([0.0]))

